import { chromium } from 'playwright'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
const errors = []
page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()) })
page.on('pageerror', (err) => errors.push(String(err)))

await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
await page.fill('.question-box', 'how much fine for using aa mirror film with 20 percent black tint')
await page.click('button.primary')
await page.waitForSelector('.response-card', { timeout: 60000 })
await page.waitForTimeout(300)
await page.screenshot({ path: 'shot_grounding_chat.png', fullPage: true })

console.log('CONSOLE_ERRORS:', JSON.stringify(errors))
await browser.close()
