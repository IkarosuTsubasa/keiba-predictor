# マイ予測双页面设计校对

- source visual truth paths:
  - `D:\keiba\new\analysis_outputs\product_design_selected_settings_option1.png`（已选定的第一种视觉方向）
  - `D:\keiba\new\analysis_outputs\product_design_user_settings_feedback.png`（用户指出重复概览与摆位问题的设置页）
  - `D:\keiba\new\analysis_outputs\product_design_user_home_before.png`（改版前的マイ予測主页）
- implementation screenshot paths:
  - `D:\keiba\new\analysis_outputs\product_design_implementation_settings_redesign.png`
  - `D:\keiba\new\analysis_outputs\product_design_implementation_home_redesign.png`
- full-view comparison evidence:
  - `D:\keiba\new\analysis_outputs\product_design_settings_redesign_comparison.png`
  - `D:\keiba\new\analysis_outputs\product_design_home_redesign_comparison.png`
- viewport: `390 × 844`
- states:
  - 设置页：地力“やや重視”、近走内容“重視”、条件适性“やや軽視”，其余为“標準”。
  - 主页：已保存上述参数，但当天比赛数据尚未准备完成。

**Findings**

- 无仍需处理的 P0、P1 或 P2 问题。
- 字体与排版：日文标题、正文、状态标签和按钮均在 390 像素宽度下保持可读；标题层级从页面标题、专属主视觉到任务卡依次收敛，没有溢出或错误换行。
- 间距与布局节奏：设置页删除了重复的“現在のバランスプロフィール”，功能标识移到主视觉顶部，主视觉直接连接参数区；主页压缩了设置摘要，使“本日分を準備中”和刷新操作都能在首屏出现。
- 色彩与视觉令牌：两页统一使用应用既有的深海军蓝、青绿色、白色与浅灰背景；高级感来自层级、图片和留白，没有新增冲突色。
- 图片质量与资产一致性：两页共用真实的数据流背景与透明品牌马图，390 像素下清晰、无黑底、无错误拉伸或遮住正文。
- 文案与内容：所有用户可见文案均为日语；主页明确表达“已保存预测设置”和“今日数据准备中”，没有展示实现方式，也没有承诺胜率或收益。
- 交互与响应：设置页的 7 个五档滑块、重置与固定保存按钮保持可用；主页的设置入口、下拉刷新和“最新情報を確認”按钮均连接原有真实逻辑。
- 可访问性可见风险：主要按钮高度不低于 48 像素，状态没有只依赖颜色表达；截图无法单独证明读屏顺序与动态字体放大，仍由 Flutter 语义测试和设备验证覆盖。

**Comparison History**

- 设置页初版：悬浮概览压住功能标识，并重复呈现滑块已经表达的信息。修复为单一主视觉区，删除整张概览卡并重新分配首屏高度。
- 主页初版：沿用普通白卡与 7 个标签，缺少与设置页一致的专属感，空状态也没有可执行动作。修复为品牌主视觉、精简的变更项摘要，以及带主动刷新的今日准备状态。
- 最终视觉复核：两张并排对比图确认页面密度、图片裁切、卡片边界、日文换行和底部导航均正常；主页刷新按钮与设置页保存按钮都在 390 × 844 状态可达。

**Implementation Checklist**

- [x] 删除无意义的重复概览卡
- [x] 修复功能标识与内容重叠
- [x] 统一设置页和主页的品牌视觉
- [x] 精简主页参数摘要，只展示实际变更项
- [x] 重做无比赛信息状态并加入真实刷新入口
- [x] 复核两页 390 × 844 截图与主要操作

**Rewarded Access Iteration**

- 已实现观看激励视频后开放一小时、提前关闭不开放、到期自动恢复锁定，以及重启后继续保留剩余时间。
- Android/iOS 的最新锁定态页面尚未完成真机截图和视觉复核；当前环境的 Flutter SDK 执行额度阻止了本轮测试与构建。

final result: blocked
