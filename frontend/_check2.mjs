import { chromium } from 'playwright'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
const errors = []
page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()) })
page.on('pageerror', (err) => errors.push(String(err)))

await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
await page.click('nav.tabs >> text=Eval')
await page.waitForSelector('.question-box')
await page.screenshot({ path: 'shot_eval_empty.png' })

await page.fill('.question-box', 'Can the officer ask for my RC?')
await page.click('button.primary')
await page.waitForSelector('.grid-2x2', { timeout: 60000 })
await page.click('button.ghost:has-text("Show ranked context chunks")')
await page.waitForTimeout(300)
await page.screenshot({ path: 'shot_eval_result.png', fullPage: true })

console.log('CONSOLE_ERRORS:', JSON.stringify(errors))
await browser.close()
