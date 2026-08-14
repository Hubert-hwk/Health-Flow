import React, { useState } from 'react';
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Form,
  Input,
  message,
  Select,
  Space,
  Table,
  Tag,
  Typography,
  Upload,
} from 'antd';
import { InboxOutlined, ReloadOutlined } from '@ant-design/icons';
import { uploadReport } from '../api.js';

const REPORT_TYPES = ['体检', '门诊', '住院', '其他'];

// 异常标记：H=偏高(红) L=偏低(橙) N=正常(绿)
export function abnormalTag(flag) {
  if (!flag) return <Tag>—</Tag>;
  const f = String(flag).toUpperCase();
  if (f === 'H' || f === 'HIGH' || f === '高') return <Tag color="red">H 偏高</Tag>;
  if (f === 'L' || f === 'LOW' || f === '低') return <Tag color="orange">L 偏低</Tag>;
  if (f === 'N' || f === 'NORMAL' || f === '正常') return <Tag color="green">N 正常</Tag>;
  return <Tag>{String(flag)}</Tag>;
}

const METRIC_COLUMNS = [
  { title: '指标', dataIndex: 'metric_name', width: 150 },
  { title: '数值', dataIndex: 'metric_value', width: 100 },
  { title: '单位', dataIndex: 'unit', width: 90 },
  { title: '参考范围', dataIndex: 'reference_range', width: 120 },
  { title: '趋势', dataIndex: 'trend', width: 70 },
  { title: '异常', dataIndex: 'abnormal_flag', width: 100, render: (v) => abnormalTag(v) },
  { title: '页码', dataIndex: 'page_number', width: 70 },
  { title: '证据原文', dataIndex: 'evidence_text', ellipsis: true },
  { title: '来源', dataIndex: 'source_id', width: 120 },
];

export default function UploadPage() {
  const [form] = Form.useForm();
  const [fileList, setFileList] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleUpload = async () => {
    const values = await form.validateFields().catch(() => null);
    if (!values) return;
    const file = fileList[0]?.originFileObj || fileList[0];
    if (!file) {
      message.warning('请先选择要上传的报告文件');
      return;
    }
    setUploading(true);
    setError('');
    try {
      const formData = new FormData();
      formData.append('patient_id', values.patient_id.trim());
      formData.append('file', file);
      if (values.report_type) formData.append('report_type', values.report_type);
      if (values.department && values.department.trim()) formData.append('department', values.department.trim());
      const data = await uploadReport(formData);
      setResult(data);
      message.success('报告解析成功');
    } catch (err) {
      setError(err.message);
      message.error(err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="page-stack">
      <Card title="上传体检报告" extra={<Typography.Text type="secondary">支持 PDF / JPG / PNG / GIF / BMP</Typography.Text>}>
        <Form form={form} layout="inline" style={{ rowGap: 16 }}>
          <Form.Item name="patient_id" label="患者编号" rules={[{ required: true, message: '请输入患者编号' }]}>
            <Input placeholder="例如 P001" style={{ width: 200 }} />
          </Form.Item>
          <Form.Item name="report_type" label="报告类型" initialValue="体检">
            <Select style={{ width: 120 }} options={REPORT_TYPES.map((t) => ({ label: t, value: t }))} />
          </Form.Item>
          <Form.Item name="department" label="科室">
            <Input placeholder="可选" style={{ width: 150 }} />
          </Form.Item>
        </Form>

        <Upload.Dragger
          style={{ marginTop: 16 }}
          accept=".pdf,.jpg,.jpeg,.png,.gif,.bmp"
          maxCount={1}
          fileList={fileList}
          beforeUpload={() => false}
          onChange={({ fileList: fl }) => setFileList(fl.slice(-1))}
          onRemove={() => setFileList([])}
        >
          <p className="ant-upload-drag-icon"><InboxOutlined /></p>
          <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
          <p className="ant-upload-hint">仅支持 PDF 或常见图片格式，单文件不超过 20MB</p>
        </Upload.Dragger>

        <Button
          type="primary"
          icon={<ReloadOutlined />}
          loading={uploading}
          onClick={handleUpload}
          style={{ marginTop: 16 }}
        >
          上传并解析
        </Button>
      </Card>

      {error && <Alert type="error" showIcon title={error} />}

      {result && (
        <Card title={`解析结果 · 报告 #${result.id}`}>
          <Descriptions column={4} size="small" bordered style={{ marginBottom: 16 }}>
            <Descriptions.Item label="患者编号">{result.patient_id}</Descriptions.Item>
            <Descriptions.Item label="报告类型">{result.report_type}</Descriptions.Item>
            <Descriptions.Item label="科室">{result.department || '—'}</Descriptions.Item>
            <Descriptions.Item label="检查日期">{result.exam_date ? new Date(result.exam_date).toLocaleString() : '—'}</Descriptions.Item>
          </Descriptions>
          <Table
            rowKey={(r) => `${r.source_id || r.metric_name || r.page_number || ''}`}
            columns={METRIC_COLUMNS}
            dataSource={result.metrics || []}
            pagination={false}
            size="small"
            scroll={{ x: 900 }}
            locale={{ emptyText: <Space direction="vertical"><Typography.Text type="secondary">未解析出指标（需要接入 VLM 解析服务）</Typography.Text></Space> }}
          />
        </Card>
      )}
    </div>
  );
}
