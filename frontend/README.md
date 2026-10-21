# 平台前端（第一阶段）

React + TypeScript + Vite 的简约前端，只通过 HTTP / SSE 与 Backend 通信。

## 启动

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

开发环境下 Vite 会把 `/api` 代理到 `http://localhost:8000`，
因此 EventSource 直接使用同源地址 `/api/v1/research/{task_id}/stream`。

## 构建与类型检查

```bash
npm run typecheck  # TypeScript 类型检查
npm run build      # 类型检查 + 生产构建（产物在 dist/）
```

## 页面

| 路由 | 页面 | 说明 |
| --- | --- | --- |
| `/` | 首页 | 新建研究（大输入框 + 开始研究）+ 基础研究历史 |
| `/research/:taskId` | 研究过程 | 左侧任务信息 / 澄清输入，右侧执行过程与最终报告 |

## 目录结构

```
frontend/src/
├── api/client.ts                 # Backend API 客户端（含 SSE 地址构造）
├── types/research.ts             # 与后端 Pydantic Schema 对齐的类型
├── hooks/useResearchStream.ts    # SSE Hook：事件 → UI 状态
├── pages/
│   ├── HomePage.tsx
│   └── ResearchPage.tsx
├── components/
│   ├── ResearchInput.tsx
│   ├── HistoryList.tsx
│   ├── ResearchTimeline.tsx
│   ├── ClarificationPanel.tsx
│   └── ReportView.tsx
└── styles/global.css
```

## 状态管理

只使用 React hooks（`useState` / `useEffect` / `useRef` / `useCallback`），未引入 Redux。
研究过程通过 `EventSource` 接收，不使用 `setInterval` 轮询。

## 交互说明

- 创建任务后跳转 `/research/:taskId`，SSE 会先回放历史事件再推送实时事件；
- 收到 `clarification_required` 时显示澄清面板，提交后调用 `/resume` 并重新连接 SSE；
- 任务完成后通过 `report_chunk` 或 `GET /report` 获取 Markdown 报告并渲染。
