import { chromium } from 'playwright'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 700 } })
const errors = []
page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()) })
page.on('pageerror', (err) => errors.push(String(err)))

await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
// Seed a bunch of history so the empty-focus default view has plenty to show
await page.evaluate(() => {
  localStorage.setItem('traffic-shield:question-history', JSON.stringify([
    'Is it mandatory to wear a helmet as a pillion rider?',
    'Can the officer ask for my RC?',
    'What is the fine for overspeeding?',
    'Is my window tint legal?',
    'Can my driving licence be seized on the spot?',
  ]))
})
await page.reload({ waitUntil: 'networkidle' })

await page.click('.question-box')
await page.waitForSelector('.suggestion-list')
const chatCount = await page.locator('.suggestion-item').count()
await page.screenshot({ path: 'shot_chat_scroll.png' })

await page.click('nav.tabs >> text=Eval')
await page.click('.question-box')
await page.waitForSelector('.suggestion-list')
const evalCount = await page.locator('.suggestion-item').count()
await page.screenshot({ path: 'shot_eval_scroll.png' })

console.log('chat suggestions count:', chatCount)
console.log('eval suggestions count:', evalCount)
console.log('CONSOLE_ERRORS:', JSON.stringify(errors))
await browser.close()
