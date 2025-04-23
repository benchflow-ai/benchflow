import antfu from '@antfu/eslint-config'

export default antfu({
  ignores: ['.next', '**/.next/**', 'node_modules', '**/node_modules/**', 'migrations/**.json'],
  formatters: false,
  react: true,
}).overrideRules({
  'style/eol-last': 'off',
})
