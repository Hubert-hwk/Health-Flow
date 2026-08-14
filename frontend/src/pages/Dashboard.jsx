import React, { useEffect, useState } from 'react';
import { getHealth, getReady } from '../api.js';

// 将状态值渲染成可读文本：对象展开为 key: value，标量直接显示
function renderValue(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'object') {
    return Object.entries(value)
      .map(([k, v]) => `${k}: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`)
      .join(' · ');
  }
  return String(value);
}

// 单张状态卡片
function StatusCard({ label, value, ok }) {
  return (
    <div className={`status-card ${ok === true ? 'ok' : ok === false ? 'bad' : ''}`}>
      <div className="status-card-label">{label}</div>
      <div className="status-card-value">{renderValue(value)}</div>
    </div>
  );
}

export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [ready, setReady] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  // 页面加载时并行请求 /health 与 /ready
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [h, r] = await Promise.all([getHealth(), getReady()]);
        if (cancelled) return;
        setHealth(h);
        setReady(r);
      } catch (e) {
        if (!cancelled) setError(e.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // 通用字段提取：优先取指定 key，其次全对象
  const field = (obj, keys) => {
    if (!obj || typeof obj !== 'object') return obj;
    for (const k of keys) {
      if (obj[k] !== undefined) return obj[k];
    }
    return obj;
  };

  const isOk = (v) => {
    if (v === true || v === 'ok' || v === 'healthy' || v === 'ready' || v === 'up') return true;
    if (v === false || v === 'down' || v === 'unhealthy' || v === 'error' || v === 'fail') return false;
    return undefined;
  };

  return (
    <div>
      {/* 后端状态卡片 */}
      <div className="status-grid">
        <StatusCard
          label="服务状态"
          value={field(health, ['status', 'service', 'state'])}
          ok={isOk(field(health, ['status', 'service', 'state']))}
        />
        <StatusCard
          label="数据库"
          value={field(ready, ['database', 'db', 'postgres'])}
          ok={isOk(field(ready, ['database', 'db', 'postgres']))}
        />
        <StatusCard
          label="Milvus 向量库"
          value={field(ready, ['milvus'])}
          ok={isOk(field(ready, ['milvus']))}
        />
        <StatusCard
          label="Neo4j 图谱"
          value={field(ready, ['neo4j'])}
          ok={isOk(field(ready, ['neo4j']))}
        />
      </div>

      {loading && <p className="muted">正在检测后端服务…</p>}
      {error && <div className="alert alert-error">无法连接后端：{error}</div>}

      {/* 原始响应查看 */}
      <div className="card">
        <h3 className="card-title">后端健康响应</h3>
        {health ? (
          <pre className="json-block">{JSON.stringify(health, null, 2)}</pre>
        ) : (
          <p className="muted">{loading ? '加载中…' : '暂无数据'}</p>
        )}
      </div>
      <div className="card">
        <h3 className="card-title">后端就绪响应</h3>
        {ready ? (
          <pre className="json-block">{JSON.stringify(ready, null, 2)}</pre>
        ) : (
          <p className="muted">{loading ? '加载中…' : '暂无数据'}</p>
        )}
      </div>
    </div>
  );
}
