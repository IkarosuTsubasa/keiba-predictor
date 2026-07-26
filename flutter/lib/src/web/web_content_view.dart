import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:webview_flutter_android/webview_flutter_android.dart';

import '../services/native_platform_bridge.dart';
import 'app_web.dart';

class WebContentView extends StatefulWidget {
  const WebContentView({
    required this.initialUri,
    required this.baseUri,
    this.onLocationChanged,
    super.key,
  });

  final Uri initialUri;
  final Uri baseUri;
  final ValueChanged<Uri>? onLocationChanged;

  @override
  State<WebContentView> createState() => WebContentViewState();
}

class WebContentViewState extends State<WebContentView> {
  static const _pullThreshold = 76.0;
  static const _automaticRetryDelays = [
    Duration(seconds: 2),
    Duration(seconds: 5),
    Duration(seconds: 10),
  ];

  late final WebViewController _controller;
  late Uri _lastRequestedUri;
  Timer? _slowLoadTimer;
  Timer? _automaticRetryTimer;
  int _automaticRetryAttempt = 0;
  int _progress = 0;
  bool _firstPageFinished = false;
  bool _showSlowLoadActions = false;
  String? _mainFrameError;
  bool _pullEnabled = false;
  double _pullDistance = 0;

  @override
  void initState() {
    super.initState();
    _lastRequestedUri = AppWeb.withAppMarker(widget.initialUri);
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0xFFF3F6F8))
      ..setNavigationDelegate(
        NavigationDelegate(
          onProgress: _onProgress,
          onPageStarted: _onPageStarted,
          onPageFinished: _onPageFinished,
          onWebResourceError: _onWebResourceError,
          onNavigationRequest: _onNavigationRequest,
          onUrlChange: (change) => _notifyLocation(change.url),
        ),
      );
    unawaited(_configureAndLoad());
    _scheduleSlowLoadActions();
  }

  Future<void> _configureAndLoad() async {
    final userAgent = NativePlatformBridge.runtimeConfig?.userAgent;
    if (userAgent != null && userAgent.isNotEmpty) {
      await _controller.setUserAgent(userAgent);
    }
    final platformController = _controller.platform;
    if (platformController is AndroidWebViewController) {
      final cookieManager = WebViewCookieManager().platform;
      if (cookieManager is AndroidWebViewCookieManager) {
        await cookieManager.setAcceptThirdPartyCookies(
          platformController,
          true,
        );
      }
    }
    await _controller.loadRequest(_lastRequestedUri);
  }

  @override
  void dispose() {
    _slowLoadTimer?.cancel();
    _automaticRetryTimer?.cancel();
    super.dispose();
  }

  Future<void> load(Uri uri) async {
    _lastRequestedUri = AppWeb.withAppMarker(uri);
    await _controller.loadRequest(_lastRequestedUri);
  }

  Future<void> reload() {
    if (_mainFrameError != null) {
      return retryAfterNetworkInterruption();
    }
    return _controller.reload();
  }

  /// Retries only when the current page never completed or its main request
  /// failed. iOS temporarily suspends networking while its first-use network
  /// permission alert is visible, but WebKit does not automatically replay the
  /// request after the user allows access.
  Future<void> retryAfterNetworkInterruption({bool resetBackoff = true}) async {
    if (_mainFrameError == null && _firstPageFinished) return;

    _automaticRetryTimer?.cancel();
    _automaticRetryTimer = null;
    if (resetBackoff) {
      _automaticRetryAttempt = 0;
    }
    if (mounted) {
      setState(() {
        _mainFrameError = null;
        _firstPageFinished = false;
        _showSlowLoadActions = false;
        _progress = 1;
      });
    }
    _scheduleSlowLoadActions();
    try {
      await _controller.loadRequest(_lastRequestedUri);
    } catch (error) {
      _slowLoadTimer?.cancel();
      if (mounted) {
        setState(() {
          _firstPageFinished = true;
          _showSlowLoadActions = false;
          _mainFrameError = error.toString();
          _progress = 100;
        });
      }
      _scheduleAutomaticRetry();
    }
  }

  Future<bool> canGoBack() => _controller.canGoBack();

  Future<void> goBack() => _controller.goBack();

  Future<void> scrollToTop() => _controller.scrollTo(0, 0);

  void _onProgress(int progress) {
    if (!mounted) return;
    setState(() => _progress = progress);
  }

  void _onPageStarted(String url) {
    final uri = Uri.tryParse(url);
    if (uri != null && AppWeb.isInternalUri(uri, widget.baseUri)) {
      _lastRequestedUri = uri;
    }
    _scheduleSlowLoadActions();
    if (!mounted) return;
    setState(() {
      _mainFrameError = null;
      _progress = math.max(_progress, 1);
    });
    _notifyLocation(url);
  }

  void _onPageFinished(String url) {
    _slowLoadTimer?.cancel();
    _automaticRetryTimer?.cancel();
    _automaticRetryTimer = null;
    _automaticRetryAttempt = 0;
    if (!mounted) return;
    setState(() {
      _firstPageFinished = true;
      _showSlowLoadActions = false;
      _progress = 100;
    });
    _notifyLocation(url);
  }

  void _onWebResourceError(WebResourceError error) {
    if (error.isForMainFrame != true || !mounted) return;
    _slowLoadTimer?.cancel();
    setState(() {
      _firstPageFinished = true;
      _showSlowLoadActions = false;
      _mainFrameError = error.description;
      _progress = 100;
    });
    _scheduleAutomaticRetry();
  }

  NavigationDecision _onNavigationRequest(NavigationRequest request) {
    if (!request.isMainFrame) {
      return NavigationDecision.navigate;
    }
    final uri = Uri.tryParse(request.url);
    if (uri == null) {
      return NavigationDecision.prevent;
    }
    if (uri.scheme == 'about') {
      return NavigationDecision.navigate;
    }
    if (AppWeb.isInternalUri(uri, widget.baseUri)) {
      final marked = AppWeb.withAppMarker(uri);
      if (marked != uri) {
        _lastRequestedUri = marked;
        unawaited(_controller.loadRequest(_lastRequestedUri));
        return NavigationDecision.prevent;
      }
      return NavigationDecision.navigate;
    }
    unawaited(_openExternal(uri));
    return NavigationDecision.prevent;
  }

  Future<void> _openExternal(Uri uri) async {
    final opened = await launchUrl(uri, mode: LaunchMode.externalApplication);
    if (opened || !mounted) return;
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('外部ページを開けませんでした。')));
  }

  void _notifyLocation(String? value) {
    final uri = value == null ? null : Uri.tryParse(value);
    if (uri != null && AppWeb.isInternalUri(uri, widget.baseUri)) {
      widget.onLocationChanged?.call(uri);
    }
  }

  void _scheduleSlowLoadActions() {
    if (_firstPageFinished) return;
    _slowLoadTimer?.cancel();
    _slowLoadTimer = Timer(const Duration(seconds: 10), () {
      if (!mounted || _firstPageFinished) return;
      setState(() => _showSlowLoadActions = true);
    });
  }

  void _scheduleAutomaticRetry() {
    if (!mounted ||
        _automaticRetryTimer != null ||
        _automaticRetryAttempt >= _automaticRetryDelays.length) {
      return;
    }
    final delay = _automaticRetryDelays[_automaticRetryAttempt];
    _automaticRetryAttempt += 1;
    _automaticRetryTimer = Timer(delay, () {
      _automaticRetryTimer = null;
      if (!mounted || _mainFrameError == null) return;
      unawaited(retryAfterNetworkInterruption(resetBackoff: false));
    });
  }

  Future<void> _onPointerDown(PointerDownEvent event) async {
    final position = await _controller.getScrollPosition();
    _pullEnabled = position.dy <= 1;
    _pullDistance = 0;
  }

  void _onPointerMove(PointerMoveEvent event) {
    if (!_pullEnabled) return;
    final next = (_pullDistance + event.delta.dy).clamp(0.0, 112.0);
    if (next == _pullDistance || !mounted) return;
    setState(() => _pullDistance = next);
  }

  void _onPointerEnd(PointerEvent event) {
    if (!_pullEnabled) return;
    final shouldReload = _pullDistance >= _pullThreshold;
    _pullEnabled = false;
    if (mounted) {
      setState(() => _pullDistance = 0);
    }
    if (shouldReload) {
      unawaited(reload());
    }
  }

  @override
  Widget build(BuildContext context) {
    final loading = _progress > 0 && _progress < 100;
    return Stack(
      children: [
        Listener(
          behavior: HitTestBehavior.translucent,
          onPointerDown: _onPointerDown,
          onPointerMove: _onPointerMove,
          onPointerUp: _onPointerEnd,
          onPointerCancel: _onPointerEnd,
          child: WebViewWidget(controller: _controller),
        ),
        if (loading)
          Align(
            alignment: Alignment.topCenter,
            child: LinearProgressIndicator(
              value: _progress / 100,
              minHeight: 3,
              color: const Color(0xFF0D6E6E),
              backgroundColor: const Color(0xFFDDEFEA),
            ),
          ),
        if (_pullDistance > 0)
          Positioned(
            top: 12,
            left: 0,
            right: 0,
            child: Center(
              child: RefreshProgressIndicator(
                value: math.min(1, _pullDistance / _pullThreshold),
                color: const Color(0xFF0D6E6E),
                backgroundColor: Colors.white,
              ),
            ),
          ),
        if (!_firstPageFinished) _buildLaunchOverlay(),
        if (_mainFrameError != null) _buildErrorOverlay(),
      ],
    );
  }

  Widget _buildLaunchOverlay() {
    return ColoredBox(
      color: const Color(0xDDF3F6F8),
      child: Center(
        child: Card(
          margin: const EdgeInsets.all(24),
          elevation: 8,
          child: Padding(
            padding: const EdgeInsets.all(26),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  '読み込み中',
                  style: TextStyle(
                    color: Color(0xFF084F4F),
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 10),
                const Text(
                  'いかいもAI競馬',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 26, fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 12),
                const Text(
                  '公開ページを読み込んでいます',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 14, color: Color(0xFF66746F)),
                ),
                if (_showSlowLoadActions) ...[
                  const SizedBox(height: 10),
                  const Text(
                    '読み込みに時間がかかっています',
                    style: TextStyle(color: Color(0xFF66746F), fontSize: 13),
                  ),
                ],
                const SizedBox(height: 18),
                const LinearProgressIndicator(
                  color: Color(0xFF0D6E6E),
                  backgroundColor: Color(0xFFE3ECE8),
                ),
                if (_showSlowLoadActions) ...[
                  const SizedBox(height: 14),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: reload,
                      child: const Text('再読み込み'),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildErrorOverlay() {
    return ColoredBox(
      color: const Color(0xFFF3F6F8),
      child: Center(
        child: Card(
          margin: const EdgeInsets.all(24),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 26),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(
                  Icons.cloud_off_outlined,
                  size: 36,
                  color: Color(0xFF66746F),
                ),
                const SizedBox(height: 14),
                const Text(
                  'ページの読み込みに失敗しました。',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 16),
                FilledButton.icon(
                  onPressed: reload,
                  icon: const Icon(Icons.refresh),
                  label: const Text('再試行'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
