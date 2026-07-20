import 'package:flutter/material.dart';

import '../models/custom_prediction.dart';

class CustomPredictionSettingsScreen extends StatefulWidget {
  const CustomPredictionSettingsScreen({
    required this.initialParameters,
    super.key,
  });

  final CustomPredictionParameters initialParameters;

  @override
  State<CustomPredictionSettingsScreen> createState() =>
      _CustomPredictionSettingsScreenState();
}

class _CustomPredictionSettingsScreenState
    extends State<CustomPredictionSettingsScreen> {
  static const _navy = Color(0xFF07111F);
  static const _accent = Color(0xFF0D7D79);
  static const _muted = Color(0xFF66746F);

  late CustomPredictionParameters _parameters;

  @override
  void initState() {
    super.initState();
    _parameters = widget.initialParameters;
  }

  void _update(CustomPredictionParameters parameters) {
    setState(() => _parameters = parameters);
  }

  void _save() {
    Navigator.of(context).pop(_parameters);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        toolbarHeight: 64,
        centerTitle: true,
        title: const Text(
          '予測設定',
          style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800),
        ),
      ),
      body: SafeArea(
        top: false,
        child: Column(
          children: [
            Expanded(
              child: ListView(
                padding: EdgeInsets.zero,
                children: [_buildHero(), _buildParameterSection()],
              ),
            ),
            _buildSaveArea(),
          ],
        ),
      ),
    );
  }

  Widget _buildHero() {
    return Container(
      height: 220,
      decoration: const BoxDecoration(
        color: _navy,
        image: DecorationImage(
          image: AssetImage('assets/images/prediction_settings_hero.png'),
          fit: BoxFit.cover,
        ),
      ),
      child: Stack(
        children: [
          Positioned(
            right: 4,
            bottom: 12,
            child: Image.asset(
              'assets/images/prediction_horse.png',
              width: 134,
              height: 126,
              fit: BoxFit.contain,
            ),
          ),
          const Positioned(
            left: 22,
            right: 126,
            top: 20,
            bottom: 18,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _CustomPredictionBadge(),
                Spacer(),
                Text(
                  'あなただけの\n予測バランス',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 22,
                    height: 1.17,
                    fontWeight: FontWeight.w800,
                    letterSpacing: -0.5,
                  ),
                ),
                SizedBox(height: 8),
                Text(
                  '重視する要素を選び、あなただけの予測スタイルへ',
                  style: TextStyle(
                    color: Color(0xFFD5E2EA),
                    fontSize: 12,
                    height: 1.45,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildParameterSection() {
    return Container(
      color: Colors.white,
      padding: const EdgeInsets.fromLTRB(18, 20, 18, 24),
      child: Column(
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '予測バランスを調整',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    SizedBox(height: 4),
                    Text(
                      '各要素の重視度を5段階で選べます',
                      style: TextStyle(color: _muted, fontSize: 12),
                    ),
                  ],
                ),
              ),
              TextButton.icon(
                onPressed: _parameters.isStandard
                    ? null
                    : () => _update(const CustomPredictionParameters()),
                icon: const Icon(Icons.restart_alt_rounded, size: 18),
                label: const Text('標準に戻す'),
              ),
            ],
          ),
          const SizedBox(height: 4),
          _PredictionParameterSlider(
            icon: Icons.fitness_center_rounded,
            title: '地力',
            description: 'レース全体で発揮できる基礎能力',
            value: _parameters.abilityLevel,
            onChanged: (value) =>
                _update(_parameters.copyWith(abilityLevel: value)),
          ),
          _PredictionParameterSlider(
            icon: Icons.trending_up_rounded,
            title: '近走内容',
            description: '直近の走りと調子の上向き度',
            value: _parameters.recentLevel,
            onChanged: (value) =>
                _update(_parameters.copyWith(recentLevel: value)),
          ),
          _PredictionParameterSlider(
            icon: Icons.track_changes_rounded,
            title: '条件適性',
            description: '距離・コース・馬場への適性',
            value: _parameters.conditionLevel,
            onChanged: (value) =>
                _update(_parameters.copyWith(conditionLevel: value)),
          ),
          _PredictionParameterSlider(
            icon: Icons.groups_rounded,
            title: '相手関係',
            description: '対戦相手と過去レースの水準',
            value: _parameters.opponentLevel,
            onChanged: (value) =>
                _update(_parameters.copyWith(opponentLevel: value)),
          ),
          _PredictionParameterSlider(
            icon: Icons.flag_rounded,
            title: '展開・騎手',
            description: '想定ペースと脚質・騎手のかみ合わせ',
            value: _parameters.paceJockeyLevel,
            onChanged: (value) =>
                _update(_parameters.copyWith(paceJockeyLevel: value)),
          ),
          _PredictionParameterSlider(
            icon: Icons.biotech_rounded,
            title: '血統適性',
            description: '今回の条件に対する血統面の適性',
            value: _parameters.pedigreeLevel,
            onChanged: (value) =>
                _update(_parameters.copyWith(pedigreeLevel: value)),
          ),
          _PredictionParameterSlider(
            icon: Icons.shield_rounded,
            title: '安定性',
            description: '不安材料と崩れるリスクの重さ',
            value: _parameters.stabilityLevel,
            onChanged: (value) =>
                _update(_parameters.copyWith(stabilityLevel: value)),
          ),
        ],
      ),
    );
  }

  Widget _buildSaveArea() {
    return Container(
      padding: const EdgeInsets.fromLTRB(18, 12, 18, 10),
      decoration: const BoxDecoration(
        color: Colors.white,
        border: Border(top: BorderSide(color: Color(0xFFE3EAEE))),
        boxShadow: [
          BoxShadow(
            color: Color(0x12020E18),
            blurRadius: 18,
            offset: Offset(0, -5),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              key: const Key('save-custom-prediction-settings'),
              onPressed: _save,
              icon: const Icon(Icons.save_outlined),
              label: const Text(
                'このバランスを保存',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800),
              ),
              style: FilledButton.styleFrom(
                backgroundColor: _accent,
                foregroundColor: Colors.white,
                minimumSize: const Size.fromHeight(52),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(28),
                ),
              ),
            ),
          ),
          const SizedBox(height: 7),
          const Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.auto_awesome_rounded, color: _muted, size: 15),
              SizedBox(width: 6),
              Flexible(
                child: Text(
                  '保存したバランスから、あなただけの答えを導き出します',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: _muted, fontSize: 11),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _CustomPredictionBadge extends StatelessWidget {
  const _CustomPredictionBadge();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
      decoration: BoxDecoration(
        color: const Color(0xFFDDF5F1),
        borderRadius: BorderRadius.circular(20),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.auto_awesome_rounded, color: Color(0xFF08736F), size: 17),
          SizedBox(width: 6),
          Text(
            'カスタム予測',
            style: TextStyle(
              color: Color(0xFF08736F),
              fontSize: 12,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }
}

class _PredictionParameterSlider extends StatelessWidget {
  const _PredictionParameterSlider({
    required this.icon,
    required this.title,
    required this.description,
    required this.value,
    required this.onChanged,
  });

  final IconData icon;
  final String title;
  final String description;
  final int value;
  final ValueChanged<int> onChanged;

  @override
  Widget build(BuildContext context) {
    const labels = ['軽視', 'やや軽視', '標準', 'やや重視', '重視'];
    final selectedLabel = CustomPredictionParameters.labelFor(value);
    return Semantics(
      label: '$title。$description',
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 9),
        decoration: const BoxDecoration(
          border: Border(bottom: BorderSide(color: Color(0xFFE6ECEF))),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 40,
              height: 40,
              alignment: Alignment.center,
              decoration: const BoxDecoration(
                color: Color(0xFFF0F6F6),
                shape: BoxShape.circle,
              ),
              child: Icon(icon, color: const Color(0xFF0D7D79), size: 20),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          title,
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Container(
                        constraints: const BoxConstraints(minWidth: 58),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 9,
                          vertical: 5,
                        ),
                        decoration: BoxDecoration(
                          color: const Color(0xFFDFF2EE),
                          borderRadius: BorderRadius.circular(18),
                        ),
                        child: Text(
                          selectedLabel,
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            color: Color(0xFF0D7D79),
                            fontSize: 10,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 2),
                  SliderTheme(
                    data: SliderTheme.of(context).copyWith(
                      trackHeight: 3,
                      activeTrackColor: const Color(0xFF0D7D79),
                      inactiveTrackColor: const Color(0xFFD6E0E4),
                      thumbColor: const Color(0xFF0D7D79),
                      overlayColor: const Color(0x220D7D79),
                      thumbShape: const RoundSliderThumbShape(
                        enabledThumbRadius: 8,
                      ),
                      overlayShape: const RoundSliderOverlayShape(
                        overlayRadius: 17,
                      ),
                      tickMarkShape: const RoundSliderTickMarkShape(
                        tickMarkRadius: 2.5,
                      ),
                      activeTickMarkColor: Colors.white,
                      inactiveTickMarkColor: const Color(0xFFB8C7CD),
                    ),
                    child: SizedBox(
                      height: 34,
                      child: Slider(
                        value: value.toDouble(),
                        min: CustomPredictionParameters.minLevel.toDouble(),
                        max: CustomPredictionParameters.maxLevel.toDouble(),
                        divisions: 4,
                        label: selectedLabel,
                        semanticFormatterCallback: (_) =>
                            '$title、$selectedLabel',
                        onChanged: (next) => onChanged(next.round()),
                      ),
                    ),
                  ),
                  Transform.translate(
                    offset: const Offset(0, -3),
                    child: Row(
                      children: [
                        for (final label in labels)
                          Expanded(
                            child: Text(
                              label,
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                color: label == selectedLabel
                                    ? const Color(0xFF0D7D79)
                                    : const Color(0xFF7B8985),
                                fontSize: 8,
                                fontWeight: label == selectedLabel
                                    ? FontWeight.w800
                                    : FontWeight.w500,
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
