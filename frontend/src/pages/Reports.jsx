import React, { useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Drawer,
  Form,
  Input,
  message,
  Popconfirm,
  Space,
  Table,
  Tag,
  Typography,
} from 'antd';
import { DeleteOutlined, EyeOutlined, SearchOutlined } from '@ant-design/icons';
import { getReports, getReport, getReportMetrics, deleteReport } from '../api.js';
import { abnormalTag } from './Upload.jsx';

export default function Reports() {
  const [form] = Form.useForm();
  const [reports, setReports] = useState(null); // null = 尚未查询
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [detailOpen, setDetailOpen] = useState(false);
  const [detail, setDetail] = useState(null);
  const [detailMetrics, setDetailMetrics] = useState([]);
  const [detailLoading, setDetailLoading] = useState(false);

  const search = async (values) => {
    const pid = (values?.patient_id || '').trim();
    if (!pid) {
      setError('请填写患者编号');
      return;
    }
    setError('');
    setLoading(true);
    try {
      const data = await getReports({ patient_id: pid, department: values?.department?.trim() || undefined });
      setReports(Array.isArray(data) ? data : (data && data.reports) || []);
    } catch (err) {
      setError(err.message);
      setReports([]);
    } finally {
      setLoading(false);
    }
  };

  const openDetail = async (report) => {
    setDetailOpen(true);
    setDetail(report);
    setDetailMetrics([]);
    setDetailLoading(true);
    try {
      const [detailData, metrics] = await Promise.all([
        getReport(report.id),
        getReportMetrics(report.id),
      ]);
      setDetail(detailData && detailData.id !== undefined ? detailData : report);
      setDetailMetrics(Array.isArray(metrics) ? metrics : (detailData && detailData.metrics) || []);
    } catch (err) {
      message.error(err.message);
    } finally {
      setDetailLoading(false);
    }
  };

  const remove = async (reportId) => {
    try {
      const data = await deleteReport(reportId);
      message.success(data?.message || '报告已删除');
      setReports((prev) => (prev || []).filter((r) => r.id !== reportId));
      if (detail?.id === reportId) setDetailOpen(false);
    } catch (err) {
      message.error(err.message);
    }
  };

  const columns = [
    { title: 'ID', dataIndex: 'id', width: 70 },
    { title: '类型', dataIndex: 'report_type', width: 90, render: (v) => <Tag>{v || '—'}</Tag> },
    {
      title: '检查日期', dataIndex: 'exam_date', width: 180,
      render: (v) => (v ? new Date(v).toLocaleString() : '—'),
    },
    { title: '科室', dataIndex: 'department', width: 110, render: (v) => v || '—' },
    {
      title: '指标数', dataIndex: 'metrics', width: 90,
      render: (ms) => (Array.isArray(ms) ? ms.length : '—'),
    },
    {
      title: '操作', key: 'actions', width: 160,
      render: (_, record) => (
        <Space>
          <Button size="small" icon={<EyeOutlined />} onClick={() => openDetail(record)}>详情</Button>
          <Popconfirm title="确定删除该报告？" onConfirm={() => remove(record.id)}>
            <Button size="small" danger icon={<DeleteOutlined />}>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div className="page-stack">
      <Card title="报告查询">
        <Form form={form} layout="inline" onFinish={search}>
          <Form.Item name="patient_id" rules={[{ required: true, message: '请输入患者编号' }]}>
            <Input placeholder="患者编号，例如 P001" style={{ width: 220 }} />
          </Form.Item>
          <Form.Item name="department">
            <Input placeholder="科室（可选）" style={{ width: 160 }} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" icon={<SearchOutlined />} loading={loading}>查询</Button>
          </Form.Item>
        </Form>
        {error && <Alert style={{ marginTop: 12 }} type="error" showIcon title={error} />}
      </Card>

      {reports !== null && (
        <Card title={`报告列表（${reports.length} 条）`}>
          <Table
            rowKey="id"
            columns={columns}
            dataSource={reports}
            pagination={{ pageSize: 10, showTotal: (t) => `共 ${t} 条` }}
            locale={{ emptyText: '该患者暂无报告' }}
          />
        </Card>
      )}

      <Drawer
        title={detail ? `报告详情 #${detail.id}` : '报告详情'}
        size="large"
        open={detailOpen}
        onClose={() => setDetailOpen(false)}
      >
        {detail && (
          <Descriptions column={2} size="small" bordered style={{ marginBottom: 16 }}>
            <Descriptions.Item label="患者编号">{detail.patient_id}</Descriptions.Item>
            <Descriptions.Item label="报告类型">{detail.report_type}</Descriptions.Item>
            <Descriptions.Item label="科室">{detail.department || '—'}</Descriptions.Item>
            <Descriptions.Item label="检查日期">{detail.exam_date ? new Date(detail.exam_date).toLocaleString() : '—'}</Descriptions.Item>
            <Descriptions.Item label="创建时间">{detail.created_at ? new Date(detail.created_at).toLocaleString() : '—'}</Descriptions.Item>
          </Descriptions>
        )}
        <Typography.Title level={5}>指标明细</Typography.Title>
        <Table
          rowKey={(r) => `${r.source_id || r.metric_name}-${r.report_id || ''}`}
          size="small"
          loading={detailLoading}
          dataSource={detailMetrics}
          pagination={false}
          scroll={{ x: 640 }}
          columns={[
            { title: '指标', dataIndex: 'metric_name', width: 130 },
            { title: '数值', dataIndex: 'metric_value', width: 90 },
            { title: '单位', dataIndex: 'unit', width: 80 },
            { title: '参考范围', dataIndex: 'reference_range', width: 110 },
            { title: '异常', dataIndex: 'abnormal_flag', width: 100, render: (v) => abnormalTag(v) },
            { title: '页码', dataIndex: 'page_number', width: 60 },
            { title: 'bbox', dataIndex: 'bbox', width: 170, render: (v) => (v ? JSON.stringify(v) : '—') },
            { title: '归一化坐标', dataIndex: 'bbox_normalized', width: 170, render: (v) => (v ? JSON.stringify(v) : '—') },
            { title: '证据原文', dataIndex: 'evidence_text', ellipsis: true },
            { title: '来源', dataIndex: 'source_id', width: 100 },
          ]}
          locale={{ emptyText: '暂无指标数据' }}
        />
      </Drawer>
    </div>
  );
}
