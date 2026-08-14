import React, { useEffect, useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  message,
  Radio,
  Select,
  Space,
  Switch,
  Table,
  Tabs,
  Tag,
  Typography,
} from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { SearchOutlined } from '@ant-design/icons';
import { getMetricAnomalies, getMetricTrend, getMetricSearch } from '../api.js';
import { abnormalTag } from './Upload.jsx';

const SEVERITY_COLOR = { 高: 'red', 中: 'orange', 低: 'default' };

export default function Metrics() {
  const [patientId, setPatientId] = useState('');
  const [days, setDays] = useState(90);

  // 异常指标
  const [anomalies, setAnomalies] = useState(null);
  const [anomalyLoading, setAnomalyLoading] = useState(false);
  const [anomalyError, setAnomalyError] = useState('');

  // 趋势
  const [metricName, setMetricName] = useState(null);
  const [trendData, setTrendData] = useState(null);
  const [trendLoading, setTrendLoading] = useState(false);

  // 搜索
  const [searchKw, setSearchKw] = useState('');
  const [searchAbnormal, setSearchAbnormal] = useState(false);
  const [searchResult, setSearchResult] = useState(null);
  const [searchLoading, setSearchLoading] = useState(false);

  const pid = patientId.trim();

  const loadAnomalies = async (autoDays = days) => {
    if (!pid) { message.warning('请先输入患者编号'); return; }
    setAnomalyLoading(true);
    setAnomalyError('');
    try {
      const data = await getMetricAnomalies({ patient_id: pid, days: autoDays });
      setAnomalies(data);
    } catch (err) {
      setAnomalyError(err.message);
      setAnomalies(null);
    } finally {
      setAnomalyLoading(false);
    }
  };

  const loadTrend = async () => {
    if (!pid) { message.warning('请先输入患者编号'); return; }
    if (!metricName) { message.warning('请选择要查询的指标'); return; }
    setTrendLoading(true);
    try {
      const data = await getMetricTrend({ patient_id: pid, metric_name: metricName, days });
      setTrendData(data);
    } catch (err) {
      message.error(err.message);
    } finally {
      setTrendLoading(false);
    }
  };

  const doSearch = async () => {
    if (!pid) { message.warning('请先输入患者编号'); return; }
    setSearchLoading(true);
    try {
      const data = await getMetricSearch({
        patient_id: pid,
        keyword: searchKw.trim() || undefined,
        abnormal_only: searchAbnormal || undefined,
        limit: 50,
      });
      setSearchResult(data && data.metrics);
    } catch (err) {
      message.error(err.message);
    } finally {
      setSearchLoading(false);
    }
  };

  // 异常结果加载后，自动把第一个异常指标填入趋势选择
  useEffect(() => {
    if (anomalies && Array.isArray(anomalies.anomalies) && anomalies.anomalies.length > 0) {
      const first = anomalies.anomalies[0].metric_name;
      setMetricName((prev) => prev || first);
    }
  }, [anomalies]);

  const anomalyOptions = [
    ...new Set((anomalies?.anomalies || []).map((a) => a.metric_name)),
  ].map((name) => ({ label: name, value: name }));

  // 趋势图数据：转成 recharts 需要的格式
  const chartData = (trendData?.data_points || [])
    .map((p) => ({
      ...p,
      date: p.exam_date ? new Date(p.exam_date).toLocaleDateString() : '—',
      value: Number(p.value),
      abnormal: String(p.abnormal_flag || '').toUpperCase() === 'H' || String(p.abnormal_flag || '').toUpperCase() === 'L',
    }))
    .filter((p) => Number.isFinite(p.value))
    .sort((a, b) => new Date(a.exam_date) - new Date(b.exam_date));

  const anomalyColumns = [
    { title: '指标名称', dataIndex: 'metric_name', width: 160 },
    { title: '数值', dataIndex: 'metric_value', width: 110 },
    { title: '单位', dataIndex: 'unit', width: 90 },
    { title: '参考范围', dataIndex: 'reference_range', width: 120 },
    { title: '异常', dataIndex: 'abnormal_flag', width: 100, render: (v) => abnormalTag(v) },
    { title: '严重程度', dataIndex: 'severity', width: 100, render: (v) => (v ? <Tag color={SEVERITY_COLOR[v] || 'default'}>{v}</Tag> : '—') },
    { title: '报告 ID', dataIndex: 'report_id', width: 90 },
    { title: '备注', dataIndex: 'note', ellipsis: true },
  ];

  const searchColumns = [
    { title: '指标名称', dataIndex: 'metric_name', width: 160 },
    { title: '数值', dataIndex: 'metric_value', width: 110 },
    { title: '单位', dataIndex: 'unit', width: 90 },
    { title: '参考范围', dataIndex: 'reference_range', width: 120 },
    { title: '异常', dataIndex: 'abnormal_flag', width: 100, render: (v) => abnormalTag(v) },
    { title: '报告 ID', dataIndex: 'report_id', width: 90 },
    { title: '检查日期', dataIndex: 'exam_date', render: (v) => (v ? new Date(v).toLocaleString() : '—') },
  ];

  return (
    <div className="page-stack">
      <Card>
        <Space wrap>
          <Input
            placeholder="患者编号，例如 P001"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            style={{ width: 220 }}
            allowClear
          />
          <Radio.Group value={days} onChange={(e) => setDays(e.target.value)} optionType="button" buttonStyle="solid">
            <Radio.Button value={30}>近 30 天</Radio.Button>
            <Radio.Button value={90}>近 90 天</Radio.Button>
            <Radio.Button value={180}>近 180 天</Radio.Button>
            <Radio.Button value={365}>近 365 天</Radio.Button>
          </Radio.Group>
        </Space>
      </Card>

      <Tabs
        items={[
          {
            key: 'anomalies',
            label: '异常指标汇总',
            children: (
              <Card
                title="异常指标"
                extra={<Button type="primary" size="small" loading={anomalyLoading} onClick={() => loadAnomalies()}>加载</Button>}
              >
                {anomalyError && <Alert type="error" showIcon title={anomalyError} style={{ marginBottom: 12 }} />}
                {anomalies && (
                  <Typography.Paragraph type="secondary">{anomalies.summary}</Typography.Paragraph>
                )}
                <Table
                  rowKey={(r) => `${r.metric_name}-${r.report_id}`}
                  columns={anomalyColumns}
                  dataSource={anomalies?.anomalies || []}
                  loading={anomalyLoading}
                  size="small"
                  pagination={{ pageSize: 10 }}
                  locale={{ emptyText: anomalies ? '暂无异常指标' : '点击「加载」查询' }}
                />
              </Card>
            ),
          },
          {
            key: 'trend',
            label: '指标趋势',
            children: (
              <Card title="趋势折线图" extra={
                <Space>
                  <Select
                    placeholder="选择指标"
                    style={{ width: 200 }}
                    value={metricName}
                    onChange={setMetricName}
                    options={anomalyOptions}
                    showSearch
                    allowClear
                    filterOption={(input, option) => (option?.label || '').toLowerCase().includes(input.toLowerCase())}
                  />
                  <Button type="primary" loading={trendLoading} onClick={loadTrend}>查询趋势</Button>
                </Space>
              }>
                {trendData && (
                  <Typography.Paragraph type="secondary">
                    共 {trendData.statistics?.count ?? 0} 个数据点 · 平均值 {trendData.statistics?.average ?? '—'} · 总趋势 {trendData.statistics?.overall_trend ?? '—'}
                  </Typography.Paragraph>
                )}
                {chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={chartData} margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#eef0f4" />
                      <XAxis dataKey="date" />
                      <YAxis domain={['auto', 'auto']} />
                      <Tooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="value"
                        name="指标数值"
                        stroke="#1677ff"
                        strokeWidth={2}
                        dot={{ r: 5, fill: '#1677ff' }}
                        activeDot={{ r: 7 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <Typography.Text type="secondary">{trendLoading ? '加载中…' : '暂无趋势数据，请先选择指标并查询'}</Typography.Text>
                )}
              </Card>
            ),
          },
          {
            key: 'search',
            label: '指标搜索',
            children: (
              <Card
                title="搜索指标"
                extra={
                  <Space>
                    <Input
                      placeholder="指标关键词（如 血糖）"
                      value={searchKw}
                      onChange={(e) => setSearchKw(e.target.value)}
                      style={{ width: 180 }}
                      allowClear
                    />
                    <span>仅异常 <Switch size="small" checked={searchAbnormal} onChange={setSearchAbnormal} /></span>
                    <Button type="primary" icon={<SearchOutlined />} loading={searchLoading} onClick={doSearch}>搜索</Button>
                  </Space>
                }
              >
                <Table
                  rowKey={(r) => `${r.metric_name}-${r.report_id}`}
                  columns={searchColumns}
                  dataSource={searchResult || []}
                  loading={searchLoading}
                  size="small"
                  pagination={{ pageSize: 10 }}
                  locale={{ emptyText: searchResult ? '无匹配结果' : '输入条件后点击「搜索」' }}
                />
              </Card>
            ),
          },
        ]}
      />
    </div>
  );
}
