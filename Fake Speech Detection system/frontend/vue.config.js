// vue.config.js

/**
 * @type {import('@vue/cli-service').ProjectOptions}
 */
module.exports = {
  assetsDir: 'static',
  lintOnSave: false,
  devServer: {
    proxy: {
    '/api': {// 匹配所有以 '/api'开头的请求路径
      target: 'http://127.0.0.1:8050',// 代理目标的基础路径
      changeOrigin: true,
      pathRewrite: {'^/api': ''}
    },
    }
  }
}
