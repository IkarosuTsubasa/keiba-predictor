import 'package:flutter/material.dart';

import '../web/web_content_view.dart';

class StaticWebPage extends StatefulWidget {
  const StaticWebPage({
    required this.title,
    required this.uri,
    required this.baseUri,
    super.key,
  });

  final String title;
  final Uri uri;
  final Uri baseUri;

  @override
  State<StaticWebPage> createState() => _StaticWebPageState();
}

class _StaticWebPageState extends State<StaticWebPage> {
  final _webKey = GlobalKey<WebContentViewState>();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        toolbarHeight: 72,
        title: Text(
          widget.title,
          style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
        ),
      ),
      body: WebContentView(
        key: _webKey,
        initialUri: widget.uri,
        baseUri: widget.baseUri,
      ),
    );
  }
}
