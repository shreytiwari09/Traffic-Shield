import { chromium } from 'playwright'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 700 } })
const errors = []
page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()) })
page.on('pageerror', (err) => errors.push(String(err)))

await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })

// Seed history via localStorage directly (faster than actually running 2 full asks)
await page.evaluate(() => {
  localStorage.setItem('traffic-shield:question-history', JSON.stringify([
    'Is it mandatory to wear a helmet as a pillion rider?',
    'Can the officer ask for my RC?',
  ]))
})

// Reload so the app is in a fresh state with seeded history
await page.reload({ waitUntil: 'networkidle' })

// Focus with EMPTY input — should show recent history + related curated suggestions
await page.click('.question-box')
await page.waitForSelector('.suggestion-list')
await page.screenshot({ path: 'shot_smart_empty.png' })

// Type something that should surface a related-but-different curated suggestion
// (helmet history should surface "fine" related suggestion via topic overlap)
await page.fill('.question-box', 'fine')
await page.waitForTimeout(200)
await page.screenshot({ path: 'shot_smart_typed.png' })

console.log('CONSOLE_ERRORS:', JSON.stringify(errors))
await browser.close()
