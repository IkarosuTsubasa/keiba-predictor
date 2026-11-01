import 'package:firebase_analytics/firebase_analytics.dart';
import 'package:firebase_core/firebase_core.dart';

Future<void> simulationEvent(
  String name, [
  Map<String, Object> parameters = const {},
]) async {
  if (Firebase.apps.isEmpty) return;
  try {
    await FirebaseAnalytics.instance.logEvent(
      name: name,
      parameters: parameters,
    );
  } on Object {
    /* Analytics is optional. */
  }
}
