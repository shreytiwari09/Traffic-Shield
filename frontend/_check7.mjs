import { chromium } from 'playwright'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 700 } })
const errors = []
page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()) })
page.on('pageerror', (err) => errors.push(String(err)))

await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
await page.click('.question-box')
await page.type('.question-box', 'helmet')
await page.waitForSelector('.suggestion-list')
await page.screenshot({ path: 'shot_suggestions.png' })

console.log('CONSOLE_ERRORS:', JSON.stringify(errors))
await browser.close()
