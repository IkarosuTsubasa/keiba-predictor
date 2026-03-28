# Amazon Affiliate 维护记录

## 当前投放位置
- 页面：`race详情页`
- 组件位置：`レース結果` 下方
- 代码入口：`frontend/src/components/RaceDetailPage.jsx`
- 展示组件：`frontend/src/components/RaceDetailAffiliateCard.jsx`
- 配置文件：`frontend/src/content/affiliateContent.js`

## 当前素材
- 图片文件：`frontend/public/affiliate/amazon-race-detail-ludo-640x360.jpg`
- 跳转链接：`https://amzn.to/4sAz0rV`

## 维护方式
以后更换 Amazon 素材时，优先只改：

1. `frontend/src/content/affiliateContent.js` 里的 `href`
2. `frontend/src/content/affiliateContent.js` 里的 `imageSrc`
3. 如有需要，再替换 `frontend/public/affiliate/` 目录下的图片文件

## 当前实现原则
- 这是一个简单图片链接位，不使用第三方广告脚本
- 保持图片原始比例，前端只做响应式缩放
- 不在这里接 Adsterra 等脚本广告，避免自动跳转问题
- 用户可见位置保持在 `レース結果` 下方，便于后续统一查找

## 如果以后要扩展
- 若要支持多个 Amazon 推荐位，优先继续走配置化方式
- 建议在 `affiliateContent.js` 中增加多个对象，不要把链接和图片写死到页面组件里
