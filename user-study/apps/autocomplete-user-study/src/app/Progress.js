import React from 'react';
import 'antd/dist/antd.css';
import { Progress } from 'antd';

export default ({ current, total }) => (
    <Progress percent={100 * current / total} />
);