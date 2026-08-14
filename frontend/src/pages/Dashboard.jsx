import React, { useEffect, useState } from 'react';
import { Alert, Badge, Card, Col, Descriptions, Row, Spin, Typography } from 'antd';
import {
  CheckCircleFilled,
  CloseCircleFilled,
  CloudServerOutlined,
  DatabaseOutlined,
  ApartmentOutlined,
  BranchesOutlined,
} from '@ant-design/icons';
import { getHealth, getReady } from '../api.js';

function statusColor(v) {
  if (v === true || v === 'ok' || v === 'healthy' || v === 'ready' || v === 'up') return 'success';
  if (v === false || v === 'down' || v === 'unhealthy' || v === 'error' || v === 'fail') return 'error';
  return 'default'; // optional_unavailable / 未知
}

function StatusCard({ title, icon, value, loading }) {
  const color = statusColor(value);
  return (
    <Card size="small" loading={loading}>
      <div className="status-card">
        <div className="status-icon" style={{ color: color === 'success' ? '#52c41a' : color === 'error' ? '#ff4d4f' : '#8c8c8c' }}>
          {icon}
        </div>
        <div>
          <div className="status-title">{title}</div>
          <div className="status-value">
            {value === undefined || value === null || value === '' ? (
              <Typography.Text type="secondary">—</Typography.Text>
            ) : (
              <Badge status={color} text={<span style={{ fontWeight: 600 }}>{String(value)}</span>} />
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}

export default function Dashboard() {
  const [health, setHealth] = useState(null);
  const [ready, setReady] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

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

  const field = (obj, keys) => {
    if (!obj || typeof obj !== 'object') return obj;
    for (const k of keys) {
      if (obj[k] !== undefined) return obj[k];
    }
    return obj;
  };

  return (
    <div className="page-stack">
      <Card>
        <Typography.Title level={5} style={{ marginTop: 0 }}>服务状态</Typography.Title>
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} lg={6}>
            <StatusCard title="服务状态" icon={<CloudServerOutlined />} value={field(health, ['status', 'service'])} loading={loading} />
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <StatusCard title="数据库" icon={<DatabaseOutlined />} value={field(ready, ['database', 'db'])} loading={loading} />
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <StatusCard title="Milvus 向量库" icon={<BranchesOutlined />} value={field(ready, ['milvus'])} loading={loading} />
          </Col>
          <Col xs={24} sm={12} lg={6}>
            <StatusCard title="Neo4j 图谱" icon={<ApartmentOutlined />} value={field(ready, ['neo4j'])} loading={loading} />
          </Col>
        </Row>

        {loading && <Spin style={{ marginTop: 16 }} />}
        {error && (
          <Alert
            style={{ marginTop: 16 }}
            type="error"
            showIcon
            title="无法连接后端"
            description={error}
          />
        )}
        {!loading && !error && (
          <Alert
            style={{ marginTop: 16 }}
            type="success"
            showIcon
            icon={<CheckCircleFilled />}
            title="后端服务正常"
            description="Milvus / Neo4j 为可选依赖，显示 optional_unavailable 表示未配置，不影响基础功能。"
          />
        )}
      </Card>

      <Row gutter={16}>
        <Col xs={24} lg={12}>
          <Card size="small" title="后端健康响应（/health）">
            {health ? (
              <Descriptions column={1} size="small" bordered>
                <Descriptions.Item label="status">{health.status ?? '—'}</Descriptions.Item>
                <Descriptions.Item label="version">{health.version ?? '—'}</Descriptions.Item>
                <Descriptions.Item label="service">{health.service ?? '—'}</Descriptions.Item>
                <Descriptions.Item label="database">{health.database ?? '—'}</Descriptions.Item>
              </Descriptions>
            ) : (
              <Typography.Text type="secondary">{loading ? '加载中…' : '暂无数据'}</Typography.Text>
            )}
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card size="small" title="后端就绪响应（/ready）">
            {ready ? (
              <Descriptions column={1} size="small" bordered>
                <Descriptions.Item label="status">
                  <Badge status={statusColor(ready.status)} text={ready.status ?? '—'} />
                </Descriptions.Item>
                <Descriptions.Item label="database">
                  <Badge status={statusColor(ready.database)} text={ready.database ?? '—'} />
                </Descriptions.Item>
                <Descriptions.Item label="milvus">
                  <Badge status={statusColor(ready.milvus)} text={ready.milvus ?? '—'} />
                </Descriptions.Item>
                <Descriptions.Item label="neo4j">
                  <Badge status={statusColor(ready.neo4j)} text={ready.neo4j ?? '—'} />
                </Descriptions.Item>
              </Descriptions>
            ) : (
              <Typography.Text type="secondary">{loading ? '加载中…' : '暂无数据'}</Typography.Text>
            )}
          </Card>
        </Col>
      </Row>

      <Card size="small" title="提示">
        <Typography.Text type="secondary">
          <CloseCircleFilled style={{ color: '#ff4d4f', marginRight: 6 }} />
          健康流仅提供信息整理与健康辅助建议，不能替代医生诊断、开处方或给出用药剂量。高风险场景请及时就医。
        </Typography.Text>
      </Card>
    </div>
  );
}
