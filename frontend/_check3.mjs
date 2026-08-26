import { chromium } from 'playwright'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
const errors = []
page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()) })
page.on('pageerror', (err) => errors.push(String(err)))

await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
await page.click('nav.tabs >> text=How It Works')
await page.waitForSelector('.arch-diagram')
await page.screenshot({ path: 'shot_arch.png', fullPage: true })

// expand a node
await page.click('.arch-node:has-text("Orchestration Service")')
await page.waitForTimeout(200)
await page.screenshot({ path: 'shot_arch_expanded.png', fullPage: true })

console.log('CONSOLE_ERRORS:', JSON.stringify(errors))
await browser.close()
