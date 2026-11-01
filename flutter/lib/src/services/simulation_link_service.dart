import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import '../config/app_config.dart';
import '../models/simulation_context.dart';
import '../screens/home_shell.dart';

class SimulationRouteParser extends RouteInformationParser<Uri> {
  const SimulationRouteParser();
  @override
  Future<Uri> parseRouteInformation(RouteInformation routeInformation) =>
      SynchronousFuture(routeInformation.uri);
  @override
  RouteInformation restoreRouteInformation(Uri configuration) =>
      RouteInformation(uri: configuration);
}

class SimulationRouter extends RouterDelegate<Uri>
    with ChangeNotifier, PopNavigatorRouterDelegateMixin<Uri> {
  SimulationRouter(this.observers);
  final List<NavigatorObserver> observers;
  final _homeKey = GlobalKey<HomeShellState>();
  @override
  final navigatorKey = GlobalKey<NavigatorState>();
  Uri _location = Uri(path: '/');
  String? _lastContext;
  DateTime? _lastReceived;
  @override
  Uri get currentConfiguration => _location;
  @override
  Future<void> setNewRoutePath(Uri configuration) async {
    final context = SimulationContext.fromUri(
      configuration,
      Uri.parse(AppConfig.baseWebUrl),
    );
    if (context == null) return;
    final identity = configuration.toString();
    if (_lastContext == identity &&
        _lastReceived != null &&
        DateTime.now().difference(_lastReceived!) <
            const Duration(seconds: 1)) {
      return;
    }
    _lastContext = identity;
    _lastReceived = DateTime.now();
    _location = configuration;
    notifyListeners();
    WidgetsBinding.instance.addPostFrameCallback(
      (_) => _homeKey.currentState?.openSimulation(context),
    );
  }

  @override
  Widget build(BuildContext context) => Navigator(
    key: navigatorKey,
    observers: observers,
    pages: [
      MaterialPage<void>(
        key: const ValueKey('home'),
        child: HomeShell(key: _homeKey),
      ),
    ],
    onDidRemovePage: (_) {},
  );
}
