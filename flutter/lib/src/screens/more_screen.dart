import 'dart:async';

import 'package:flutter/material.dart';

import '../config/app_config.dart';
import '../services/notification_settings.dart';
import '../services/push_notification_service.dart';
import '../web/app_web.dart';
import 'static_web_page.dart';

class MoreScreen extends StatefulWidget {
  const MoreScreen({this.showNotificationSettings, super.key});

  final bool? showNotificationSettings;

  @override
  State<MoreScreen> createState() => _MoreScreenState();
}

class _MoreScreenState extends State<MoreScreen> {
  static const _pages = [
    _MoreEntry(
      title: 'プライバシーポリシー',
      description: '個人情報の取り扱い方針を確認できます',
      path: 'privacy',
    ),
    _MoreEntry(
      title: '利用規約',
      description: 'サービスの利用条件と禁止事項を確認できます',
      path: 'terms',
    ),
    _MoreEntry(
      title: '免責事項',
      description: '予想情報と掲載データの注意事項を確認できます',
      path: 'disclaimer',
    ),
    _MoreEntry(
      title: 'お問い合わせ',
      description: 'ご意見やご連絡先を確認できます',
      path: 'contact',
    ),
  ];

  NotificationSettingsRepository? _settingsRepository;
  NotificationSettings? _settings;

  @override
  void initState() {
    super.initState();
    if (widget.showNotificationSettings ?? AppConfig.notificationsEnabled) {
      _settingsRepository = NotificationSettingsRepository();
      unawaited(_loadSettings());
    }
  }

  Future<void> _loadSettings() async {
    NotificationSettings settings;
    try {
      settings = await _settingsRepository!.load();
    } catch (_) {
      settings = const NotificationSettings();
    }
    if (!mounted) return;
    setState(() => _settings = settings);
  }

  void _updateSettings(NotificationSettings settings) {
    final shouldRequestPermission =
        _settings?.enabled == false && settings.enabled;
    setState(() => _settings = settings);
    unawaited(_settingsRepository!.save(settings));
    if (shouldRequestPermission) {
      unawaited(
        PushNotificationService.instance.requestPermissionFromSettings(),
      );
    }
  }

  void _openPage(_MoreEntry page) {
    final baseUri = Uri.parse(AppConfig.baseWebUrl);
    final uri = AppWeb.normalizeInAppUrl(
      '${AppConfig.baseWebUrl}/${page.path}',
      AppConfig.baseWebUrl,
    );
    if (uri == null) return;
    Navigator.of(context).push<void>(
      MaterialPageRoute(
        builder: (_) =>
            StaticWebPage(title: page.title, uri: uri, baseUri: baseUri),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(18, 18, 18, 28),
      children: [
        if (widget.showNotificationSettings ??
            AppConfig.notificationsEnabled) ...[
          _buildNotificationCard(),
          const SizedBox(height: 12),
        ],
        for (final page in _pages) ...[
          _buildPageCard(page),
          const SizedBox(height: 12),
        ],
      ],
    );
  }

  Widget _buildNotificationCard() {
    final settings = _settings;
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '通知設定',
              style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 6),
            const Text(
              '予測完了と日報更新のお知らせを受け取ります',
              style: TextStyle(color: Color(0xFF66746F), fontSize: 13),
            ),
            const SizedBox(height: 14),
            if (settings == null)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 26),
                child: Center(
                  child: CircularProgressIndicator(color: Color(0xFF0D6E6E)),
                ),
              )
            else ...[
              _NotificationSwitch(
                title: '通知を受け取る',
                description: 'アプリからのお知らせを表示します',
                value: settings.enabled,
                onChanged: (value) =>
                    _updateSettings(settings.copyWith(enabled: value)),
              ),
              const Divider(height: 13, color: Color(0xFFE5EAEE)),
              _NotificationSwitch(
                title: '中央競馬',
                description: '中央開催の予測通知を受け取ります',
                value: settings.centralEnabled,
                enabled: settings.enabled,
                onChanged: (value) =>
                    _updateSettings(settings.copyWith(centralEnabled: value)),
              ),
              _NotificationSwitch(
                title: '地方競馬',
                description: '地方開催の予測通知を受け取ります',
                value: settings.localEnabled,
                enabled: settings.enabled,
                onChanged: (value) =>
                    _updateSettings(settings.copyWith(localEnabled: value)),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildPageCard(_MoreEntry page) {
    return Card(
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: () => _openPage(page),
        child: ConstrainedBox(
          constraints: const BoxConstraints(minHeight: 82),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 15),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        page.title,
                        style: const TextStyle(
                          fontSize: 17,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        page.description,
                        style: const TextStyle(
                          color: Color(0xFF66746F),
                          fontSize: 13,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                const Icon(Icons.chevron_right, color: Color(0xFF8C9692)),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _NotificationSwitch extends StatelessWidget {
  const _NotificationSwitch({
    required this.title,
    required this.description,
    required this.value,
    required this.onChanged,
    this.enabled = true,
  });

  final String title;
  final String description;
  final bool value;
  final bool enabled;
  final ValueChanged<bool> onChanged;

  @override
  Widget build(BuildContext context) {
    return Opacity(
      opacity: enabled ? 1 : 0.48,
      child: SwitchListTile(
        contentPadding: EdgeInsets.zero,
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.w600)),
        subtitle: Text(description),
        value: value,
        onChanged: enabled ? onChanged : null,
      ),
    );
  }
}

class _MoreEntry {
  const _MoreEntry({
    required this.title,
    required this.description,
    required this.path,
  });

  final String title;
  final String description;
  final String path;
}
