import React from 'react';
import { createRoot } from 'react-dom/client';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import App from './App.jsx';
import './styles.css';

// Ant Design 全局配置：中文语言包 + 医疗蓝主题
createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: '#1677ff',
          colorInfo: '#1677ff',
          borderRadius: 8,
          fontSize: 14,
        },
        components: {
          Layout: { siderBg: '#ffffff', headerBg: '#ffffff', bodyBg: '#f0f4fa' },
          Menu: { itemSelectedBg: '#e6f4ff', itemSelectedColor: '#1677ff' },
        },
      }}
    >
      <App />
    </ConfigProvider>
  </React.StrictMode>
);
