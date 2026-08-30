import { chromium } from 'playwright'

const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
const errors = []
page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()) })
page.on('pageerror', (err) => errors.push(String(err)))

await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' })
await page.click('.question-box')
await page.type('.question-box', 'helmet')
await page.waitForSelector('.suggestion-list')
await page.click('.suggestion-item')
// Confirm it filled the box AND actually submitted (pipeline trace should appear)
const value = await page.inputValue('.question-box')
console.log('QUESTION_BOX_VALUE:', value)
await page.waitForSelector('.pipeline-trace', { timeout: 10000 }).then(
  () => console.log('SUBMITTED: yes, pipeline trace appeared'),
  () => console.log('SUBMITTED: no, pipeline trace never appeared')
)
await page.waitForSelector('.response-card', { timeout: 60000 }).then(
  () => console.log('COMPLETED: yes, response card appeared'),
  () => console.log('COMPLETED: no, response card never appeared')
)
await page.screenshot({ path: 'shot_suggestion_flow.png', fullPage: true })

console.log('CONSOLE_ERRORS:', JSON.stringify(errors))
await browser.close()
