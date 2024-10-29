// 2. 修复警告面板中的Row
class _MonitoringAnalysisScreenState extends State<MonitoringAnalysisScreen> {
  // ... 其他代码保持不变 ...

  Widget _buildWarningsPanel() {
    final warnings = _analysisData?['warnings'] as List? ?? [];

    return Card(
      color: warnings.isEmpty ? null : Colors.red[50],
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.warning,
                  color: warnings.isEmpty ? Colors.green : Colors.red,
                ),
                const SizedBox(width: 8),
                Expanded(  // 添加 Expanded
                  child: Text(
                    warnings.isEmpty ? '无异常' : '异常警告',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: warnings.isEmpty ? Colors.green : Colors.red,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
            if (warnings.isNotEmpty) ...[
              const SizedBox(height: 8),
              ConstrainedBox(
                constraints: const BoxConstraints(maxHeight: 200),
                child: SingleChildScrollView(
                  child: Column(
                    children: warnings.map((warning) => Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Row(
                        children: [
                          const Icon(Icons.error_outline,
                              color: Colors.red,
                              size: 16
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              warning.toString(),
                              overflow: TextOverflow.ellipsis,
                              maxLines: 2,
                            ),
                          ),
                        ],
                      ),
                    )).toList(),
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  // 3. 修复数据卡片中的Row
  Widget _buildAnalyticsCard() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '实时数据',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            SingleChildScrollView(  // 添加水平滚动
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  _buildDataItem(
                    '总人数',
                    _analysisData?['total_count']?.toString() ?? '0',
                    Icons.people,
                  ),
                  const SizedBox(width: 16),
                  _buildDataItem(
                    '密度',
                    '${((_analysisData?['crowd_density'] ?? 0) * 100).toStringAsFixed(1)}%',
                    Icons.density_medium,
                  ),
                  const SizedBox(width: 16),
                  _buildDataItem(
                    '速度',
                    '${(_analysisData?['velocity'] ?? 0).toStringAsFixed(1)} m/s',
                    Icons.speed,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDataItem(String label, String value, IconData icon) {
    return Container(
      constraints: const BoxConstraints(minWidth: 100),  // 设置最小宽度
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 24, color: Colors.blue),
          const SizedBox(height: 8),
          Text(
            label,
            style: const TextStyle(fontSize: 14),
          ),
          Text(
            value,
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}