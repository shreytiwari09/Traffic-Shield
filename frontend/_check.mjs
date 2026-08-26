import { chromium } from 'playwright'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
const errors = []
page.on('console', (msg) => {
  if (msg.type() === 'error') errors.push(msg.text())
})
page.on('pageerror', (err) => errors.push(String(err)))

await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
await page.screenshot({ path: 'shot_chat_empty.png' })

// Fill and ask
await page.fill('textarea.question-box', 'Can the officer ask for my RC?')
await page.click('button.primary')
await page.waitForSelector('.response-card', { timeout: 45000 })
await page.screenshot({ path: 'shot_chat_response.png', fullPage: true })

// Click "show original legal text" to reveal evidence view
const showBtn = page.locator('button.ghost', { hasText: 'Show original legal text' })
if (await showBtn.count()) {
  await showBtn.click()
  await page.waitForTimeout(300)
  await page.screenshot({ path: 'shot_chat_evidence.png', fullPage: true })
}

// Rights library tab
await page.click('nav.tabs >> text=Rights Library')
await page.waitForSelector('.category-grid', { timeout: 10000 })
await page.screenshot({ path: 'shot_library.png' })
await page.click('.category-card >> nth=0')
await page.waitForSelector('.section-list', { timeout: 15000 })
await page.screenshot({ path: 'shot_library_sections.png', fullPage: true })

console.log('CONSOLE_ERRORS:', JSON.stringify(errors))
await browser.close()
