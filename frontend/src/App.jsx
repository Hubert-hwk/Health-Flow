import React, { useState } from 'react';
import { Layout, Menu, Typography, Tag } from 'antd';
import {
  DashboardOutlined,
  UploadOutlined,
  FileTextOutlined,
  LineChartOutlined,
  MessageOutlined,
  ApartmentOutlined,
} from '@ant-design/icons';
import Dashboard from './pages/Dashboard.jsx';
import Upload from './pages/Upload.jsx';
import Reports from './pages/Reports.jsx';
import Metrics from './pages/Metrics.jsx';
import Chat from './pages/Chat.jsx';
import KnowledgeGraph from './pages/KnowledgeGraph.jsx';

const { Sider, Header, Content } = Layout;

const NAV_ITEMS = [
  { key: 'dashboard', label: '首页 / 概览', icon: <DashboardOutlined /> },
  { key: 'upload', label: '报告上传', icon: <UploadOutlined /> },
  { key: 'reports', label: '报告列表', icon: <FileTextOutlined /> },
  { key: 'metrics', label: '指标分析', icon: <LineChartOutlined /> },
  { key: 'chat', label: '智能问答', icon: <MessageOutlined /> },
  { key: 'kg', label: '知识图谱', icon: <ApartmentOutlined /> },
];

export default function App() {
  const [page, setPage] = useState('dashboard');
  const current = NAV_ITEMS.find((i) => i.key === page);

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
    <Layout style={{ minHeight: '100vh' }}>
      <Sider width={220} theme="light" style={{ borderRight: '1px solid #eef0f4' }}>
        <div className="brand">
          <span className="brand-logo">💙</span>
          <div>
            <div className="brand-name">健康流</div>
            <div className="brand-sub">HealthFlow</div>
          </div>
        </div>
        <Menu
          mode="inline"
          selectedKeys={[page]}
          onClick={({ key }) => setPage(key)}
          items={NAV_ITEMS}
          style={{ borderInlineEnd: 'none' }}
        />
      </Sider>
      <Layout>
        <Header
          style={{
            padding: '0 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderBottom: '1px solid #eef0f4',
          }}
        >
          <Typography.Title level={4} style={{ margin: 0 }}>
            {current?.label}
          </Typography.Title>
          <Tag color="blue" style={{ marginRight: 0 }}>后端服务：localhost:8080</Tag>
        </Header>
        <Content style={{ padding: 24 }}>{renderPage()}</Content>
      </Layout>
    </Layout>
  );
}
