import { chromium } from 'playwright'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
const errors = []
page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()) })
page.on('pageerror', (err) => errors.push(String(err)))

await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
await page.click('nav.tabs >> text=Eval')
await page.fill('.question-box', 'how much fine for using aa mirror film with 20 percent black tint')
await page.click('button.primary')
await page.waitForSelector('.grid-2x2', { timeout: 90000 })
await page.waitForTimeout(500)
await page.screenshot({ path: 'shot_eval5.png', fullPage: true })

console.log('CONSOLE_ERRORS:', JSON.stringify(errors))
await browser.close()
