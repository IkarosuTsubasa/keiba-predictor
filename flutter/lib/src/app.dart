import 'package:flutter/material.dart';

import 'services/simulation_link_service.dart';

class KeibaApp extends StatefulWidget {
  const KeibaApp({this.navigatorObservers = const [], super.key});

  final List<NavigatorObserver> navigatorObservers;

  static const _chrome = Color(0xFF07111F);
  static const _accent = Color(0xFF0D6E6E);
  static const _background = Color(0xFFF3F6F8);

  static ThemeData get theme {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: _accent,
      brightness: Brightness.light,
      surface: Colors.white,
    );

    return ThemeData(
      useMaterial3: true,
      colorScheme: colorScheme,
      scaffoldBackgroundColor: _background,
      appBarTheme: const AppBarTheme(
        backgroundColor: _chrome,
        centerTitle: false,
        foregroundColor: Colors.white,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
      ),
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: Colors.white,
        indicatorColor: const Color(0xFFDDEFEA),
        labelTextStyle: WidgetStateProperty.resolveWith(
          (states) => TextStyle(
            color: states.contains(WidgetState.selected)
                ? _accent
                : const Color(0xFF66746F),
            fontSize: 12,
            fontWeight: states.contains(WidgetState.selected)
                ? FontWeight.w700
                : FontWeight.w500,
          ),
        ),
        iconTheme: WidgetStateProperty.resolveWith(
          (states) => IconThemeData(
            color: states.contains(WidgetState.selected)
                ? _accent
                : const Color(0xFF8C9692),
          ),
        ),
      ),
      cardTheme: const CardThemeData(
        color: Colors.white,
        surfaceTintColor: Colors.transparent,
        elevation: 0,
        margin: EdgeInsets.zero,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.all(Radius.circular(16)),
          side: BorderSide(color: Color(0xFFE5EAEE)),
        ),
      ),
    );
  }

  @override
  State<KeibaApp> createState() => _KeibaAppState();
}

class _KeibaAppState extends State<KeibaApp> {
  late final _router = SimulationRouter(widget.navigatorObservers);
  @override
  void dispose() {
    _router.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'いかいもAI競馬',
      debugShowCheckedModeBanner: false,
      theme: KeibaApp.theme,
      routerDelegate: _router,
      routeInformationParser: const SimulationRouteParser(),
    );
  }
}
