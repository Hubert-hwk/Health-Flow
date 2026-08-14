import React, { useState } from 'react';
import Dashboard from './pages/Dashboard.jsx';
import Upload from './pages/Upload.jsx';
import Reports from './pages/Reports.jsx';
import Metrics from './pages/Metrics.jsx';
import Chat from './pages/Chat.jsx';
import KnowledgeGraph from './pages/KnowledgeGraph.jsx';

// 侧边栏导航配置：key 用于切换页面，label 为中文显示名
const NAV_ITEMS = [
  { key: 'dashboard', label: '首页 / 概览', icon: '🏠' },
  { key: 'upload', label: '报告上传', icon: '📤' },
  { key: 'reports', label: '报告列表', icon: '📋' },
  { key: 'metrics', label: '指标分析', icon: '📈' },
  { key: 'chat', label: '智能问答', icon: '💬' },
  { key: 'kg', label: '知识图谱', icon: '🧠' },
];

export default function App() {
  const [page, setPage] = useState('dashboard');

  // 根据当前选中项渲染对应页面
  const renderPage = () => {
    switch (page) {
      case 'dashboard': return <Dashboard />;
      case 'upload': return <Upload />;
      case 'reports': return <Reports />;
      case 'metrics': return <Metrics />;
      case 'chat': return <Chat />;
      case 'kg': return <KnowledgeGraph />;
      default: return <Dashboard />;
    }
  };

  return (
    <div className="app-layout">
      {/* 左侧边栏 */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="brand-logo">💙</span>
          <div>
            <div className="brand-name">健康流</div>
            <div className="brand-sub">HealthFlow</div>
          </div>
        </div>
        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.key}
              className={`nav-item ${page === item.key ? 'active' : ''}`}
              onClick={() => setPage(item.key)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">医疗智能辅助平台 v1.0</div>
      </aside>

      {/* 右侧主内容区 */}
      <main className="main-content">
        <header className="topbar">
          <h1 className="page-title">
            {NAV_ITEMS.find((i) => i.key === page)?.label || ''}
          </h1>
          <span className="topbar-hint">后端服务：localhost:8080</span>
        </header>
        <div className="content-body">{renderPage()}</div>
      </main>
    </div>
  );
}
