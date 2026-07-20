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
  static const _accent = Color(0xFF0D6E6E);
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
      appBar: AppBar(
        titleSpacing: 0,
        title: const Text(
          '予測設定',
          style: TextStyle(fontSize: 21, fontWeight: FontWeight.w700),
        ),
      ),
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(18, 18, 18, 24),
                children: [
                  _buildIntroductionCard(),
                  const SizedBox(height: 12),
                  _buildParameterCard(),
                ],
              ),
            ),
            Container(
              padding: const EdgeInsets.fromLTRB(18, 12, 18, 14),
              decoration: const BoxDecoration(
                color: Colors.white,
                border: Border(top: BorderSide(color: Color(0xFFE5EAEE))),
              ),
              child: SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  key: const Key('save-custom-prediction-settings'),
                  onPressed: _save,
                  icon: const Icon(Icons.check_rounded),
                  label: const Text('保存して予測を見る'),
                  style: FilledButton.styleFrom(
                    minimumSize: const Size.fromHeight(52),
                    textStyle: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildIntroductionCard() {
    return const Card(
      child: Padding(
        padding: EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.tune_rounded, color: _accent),
                SizedBox(width: 9),
                Expanded(
                  child: Text(
                    '重視する要素を設定',
                    style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
                  ),
                ),
              ],
            ),
            SizedBox(height: 8),
            Text(
              'ここで保存した設定を、マイ予測のすべてのレースに使用します。',
              style: TextStyle(color: _muted, height: 1.55),
            ),
            SizedBox(height: 6),
            Text(
              '各項目は相対的な重みです。強めた要素ほど、ほかの要素より予測に反映されます。',
              style: TextStyle(color: _muted, fontSize: 13, height: 1.5),
            ),
            SizedBox(height: 6),
            Text(
              'すべて同じ強さにすると標準と同じバランスになり、保存後にまとめて計算します。',
              style: TextStyle(color: _muted, fontSize: 13, height: 1.5),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildParameterCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(18, 17, 18, 14),
        child: Column(
          children: [
            Row(
              children: [
                const Expanded(
                  child: Text(
                    '7つの予測バランス',
                    style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700),
                  ),
                ),
                TextButton.icon(
                  onPressed: _parameters.isStandard
                      ? null
                      : () => _update(const CustomPredictionParameters()),
                  icon: const Icon(Icons.restart_alt, size: 19),
                  label: const Text('標準に戻す'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            _PredictionParameterSlider(
              title: '地力',
              description: 'レース全体で発揮できる基礎能力',
              value: _parameters.abilityLevel,
              onChanged: (value) =>
                  _update(_parameters.copyWith(abilityLevel: value)),
            ),
            _PredictionParameterSlider(
              title: '近走内容',
              description: '直近の走りと調子の上向き度',
              value: _parameters.recentLevel,
              onChanged: (value) =>
                  _update(_parameters.copyWith(recentLevel: value)),
            ),
            _PredictionParameterSlider(
              title: '条件適性',
              description: '距離・コース・馬場への適性',
              value: _parameters.conditionLevel,
              onChanged: (value) =>
                  _update(_parameters.copyWith(conditionLevel: value)),
            ),
            _PredictionParameterSlider(
              title: '相手関係',
              description: '対戦相手と過去レースの水準',
              value: _parameters.opponentLevel,
              onChanged: (value) =>
                  _update(_parameters.copyWith(opponentLevel: value)),
            ),
            _PredictionParameterSlider(
              title: '展開・騎手',
              description: '想定ペースと脚質・騎手のかみ合わせ',
              value: _parameters.paceJockeyLevel,
              onChanged: (value) =>
                  _update(_parameters.copyWith(paceJockeyLevel: value)),
            ),
            _PredictionParameterSlider(
              title: '血統適性',
              description: '今回の条件に対する血統面の適性',
              value: _parameters.pedigreeLevel,
              onChanged: (value) =>
                  _update(_parameters.copyWith(pedigreeLevel: value)),
            ),
            _PredictionParameterSlider(
              title: '安定性',
              description: '不安材料と崩れるリスクの重さ',
              value: _parameters.stabilityLevel,
              onChanged: (value) =>
                  _update(_parameters.copyWith(stabilityLevel: value)),
            ),
          ],
        ),
      ),
    );
  }
}

class _PredictionParameterSlider extends StatelessWidget {
  const _PredictionParameterSlider({
    required this.title,
    required this.description,
    required this.value,
    required this.onChanged,
  });

  final String title;
  final String description;
  final int value;
  final ValueChanged<int> onChanged;

  @override
  Widget build(BuildContext context) {
    final label = CustomPredictionParameters.labelFor(value);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      description,
                      style: const TextStyle(
                        color: Color(0xFF66746F),
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 10),
              Container(
                constraints: const BoxConstraints(minWidth: 68),
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: const Color(0xFFDDEFEA),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  label,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: Color(0xFF0D6E6E),
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),
          Slider(
            value: value.toDouble(),
            min: CustomPredictionParameters.minLevel.toDouble(),
            max: CustomPredictionParameters.maxLevel.toDouble(),
            divisions: 4,
            label: label,
            semanticFormatterCallback: (_) => '$title、$label',
            onChanged: (next) => onChanged(next.round()),
          ),
        ],
      ),
    );
  }
}
