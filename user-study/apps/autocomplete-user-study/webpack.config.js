const MonacoWebpackPlugin = require('monaco-editor-webpack-plugin');
const nxConfig = require('@nrwl/react/plugins/webpack');

module.exports = (config) => {
  config = nxConfig(config);

  if (!config.plugins) {
    config.plugins = [];
  }

  config.plugins.push(
    new MonacoWebpackPlugin({
      languages: ['python']
    }));

  return config;
};