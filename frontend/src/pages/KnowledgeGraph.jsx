import React, { useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Empty,
  Input,
  message,
  Space,
  Tag,
  Typography,
} from 'antd';
import { ApartmentOutlined, SearchOutlined } from '@ant-design/icons';
import { kgDepartment } from '../api.js';

export default function KnowledgeGraph() {
  const [symptom, setSymptom] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const search = async () => {
    const s = symptom.trim();
    if (!s) { message.warning('请输入症状关键词'); return; }
    setError('');
    setLoading(true);
    try {
      const data = await kgDepartment(s);
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const hasResult = result && result.department;

  return (
    <div className="page-stack">
      <Card
        title={<Space><ApartmentOutlined /> 症状 → 科室 知识图谱查询</Space>}
        extra={
          <Space>
            <Input
              placeholder="输入症状，例如 胸痛、头晕、发热"
              value={symptom}
              onChange={(e) => setSymptom(e.target.value)}
              style={{ width: 260 }}
              allowClear
              onPressEnter={search}
            />
            <Button type="primary" icon={<SearchOutlined />} loading={loading} onClick={search}>查询</Button>
          </Space>
        }
      >
        {error && <Alert type="error" showIcon title={error} style={{ marginBottom: 12 }} />}

        {hasResult ? (
          <div className="kg-visual">
            <div className="kg-node kg-symptom">
              <Typography.Text strong>{result.symptom}</Typography.Text>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>症状</Typography.Text>
            </div>
            <div className="kg-arrow">→</div>
            <div className="kg-node kg-dept">
              <Typography.Text strong>{result.department}</Typography.Text>
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>建议就诊科室</Typography.Text>
            </div>
          </div>
        ) : (
          <Empty
            description={
              <Typography.Text type="secondary">
                {result ? '图谱中暂无该症状记录（需初始化 Neo4j 并加载本体数据）' : '输入症状查询建议就诊科室'}
              </Typography.Text>
            }
          />
        )}
      </Card>

      {hasResult && (
        <Card size="small" title="查询详情">
          <Descriptions column={2} size="small" bordered>
            <Descriptions.Item label="症状">{result.symptom}</Descriptions.Item>
            <Descriptions.Item label="建议科室">
              <Tag color="blue">{result.department}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="数据来源">{result.source || '—'}</Descriptions.Item>
          </Descriptions>
        </Card>
      )}
    </div>
  );
}
