/** @type {import('tailwindcss').Config} */
const defaultTheme = require('tailwindcss/defaultTheme');

export default {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  theme: {
    extend: {
      fontFamily: {
        inter: ['"Inter"', ...defaultTheme.fontFamily.sans]
      }
    }
  },
  plugins: [],
}

