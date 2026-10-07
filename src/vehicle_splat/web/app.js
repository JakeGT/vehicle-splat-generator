// The page: give a CAP id, see the van's Imagin renders, make its splat and follow
// the job. Plain modules, no build; the API is the service's own (api.py).

const $ = (id) => document.getElementById(id)
const STAGE_NAMES = {
  frames: 'Frames',
  camera: 'Camera',
  views: 'Other views',
  training: 'Training',
  preview: 'Preview',
  export: 'Files',
}
const ICONS = { waiting: '○', running: '◐', done: '✓', skipped: '–', failed: '✗' }
const POLL_MS = 1500

let van = null
let pollTimer = 0

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body ? { 'Content-Type': 'application/json' } : {},
    ...options,
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail?.[0]?.msg ?? body.detail ?? `${response.status} ${response.statusText}`)
  return body
}

function element(tag, properties = {}, ...children) {
  const node = Object.assign(document.createElement(tag), properties)
  node.append(...children.filter((child) => child !== null && child !== undefined))
  return node
}

// ── The GPU ──────────────────────────────────────────────────────────────────

async function showGpu() {
  const chip = $('gpu')
  try {
    const health = await api('/api/health')
    chip.textContent = health.gpu ? `GPU: ${health.gpu}` : 'No GPU visible: training will fail'
    chip.className = `chip ${health.gpu ? 'good' : 'bad'}`
    $('publish').dataset.enabled = health.portalPublishing ? 'yes' : ''
  } catch {
    chip.textContent = 'The service is not answering'
    chip.className = 'chip bad'
  }
}

// ── The van ──────────────────────────────────────────────────────────────────

function figure(url, caption, { repeat = false } = {}) {
  return element('figure', { className: repeat ? 'repeat' : '' },
    element('img', { src: url, alt: caption.textContent ?? '', loading: 'lazy' }),
    caption)
}

async function showVan(capId, paintId) {
  const section = $('van')
  section.hidden = false
  $('vanTitle').textContent = `CAP ${capId}${paintId ? `, paint ${paintId}` : ''}`
  $('vanStatus').textContent = 'Fetching its renders from Imagin…'
  $('spin').replaceChildren()
  $('views').replaceChildren()
  $('show').disabled = true
  // Until this van has loaded there is nothing to generate: the last one's
  // renders are gone from the page, so it must not be what a job is made of.
  van = null
  $('start').disabled = true
  try {
    const query = paintId ? `?paintId=${encodeURIComponent(paintId)}` : ''
    van = await api(`/api/vehicles/${capId}${query}`)
  } catch (error) {
    $('vanStatus').textContent = error.message
    return
  } finally {
    $('show').disabled = false
  }
  $('start').disabled = false
  $('vanStatus').textContent = ''
  $('spinCount').textContent = `(${van.spin.length} renders)`
  $('spin').replaceChildren(...van.spin.map(({ angle, url }) => figure(url, element('figcaption', {}, `${angle}`))))
  const own = van.views.filter((view) => view.duplicateOf === null)
  $('viewCount').textContent =
    `(${own.length} of their own; ${van.views.length - own.length} codes only repeat code 1)`
  $('views').replaceChildren(...van.views.map((view) => {
    const caption = element('figcaption', {}, `code ${view.code}`,
      view.duplicateOf !== null ? element('span', { className: 'badge dim', textContent: `= ${view.duplicateOf}` }) : null,
      view.duplicateOf === null && view.expectedFromAbove ? element('span', { className: 'badge', textContent: 'from above' }) : null)
    return figure(view.url, caption, { repeat: view.duplicateOf !== null })
  }))
  localStorage.setItem('capId', capId)
}

$('lookup').addEventListener('submit', (event) => {
  event.preventDefault()
  showVan($('capId').value.trim(), $('paintId').value.trim())
})

$('generate').addEventListener('submit', async (event) => {
  event.preventDefault()
  if (!van) return
  $('start').disabled = true
  try {
    const job = await api('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({
        capId: van.capId,
        paintId: van.paintId,
        extraViews: new FormData(event.target).get('extraViews'),
        steps: Number($('steps').value),
        maxSplats: Number($('maxSplats').value),
      }),
    })
    location.hash = `job=${job.id}`
    follow(job.id)
    listJobs()
  } catch (error) {
    alert(error.message)
  } finally {
    $('start').disabled = !van
  }
})

// ── A job ────────────────────────────────────────────────────────────────────

function duration(seconds) {
  if (seconds === null || seconds === undefined) return ''
  return seconds < 90 ? `${Math.round(seconds)} s` : `${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`
}

function renderJob(job) {
  $('job').hidden = false
  const spec = job.spec
  $('jobTitle').textContent = `CAP ${spec.cap_id}${spec.paint_id ? `, paint ${spec.paint_id}` : ''} — ${job.id}`
  $('jobStatus').textContent = job.status
  $('jobStatus').className = `pill ${job.status}`
  $('stages').replaceChildren(...job.stages.map((stage) => element('li', { className: stage.status },
    element('span', { className: 'icon', textContent: ICONS[stage.status] ?? '○' }),
    element('strong', { textContent: STAGE_NAMES[stage.name] ?? stage.name }),
    element('span', { className: 'muted', textContent: stage.detail ?? '' }),
    element('span', { className: 'muted', textContent: duration(stage.seconds) }))))

  const training = job.training
  $('trainingBox').hidden = !training
  if (training) {
    const share = training.step / training.steps
    $('bar').style.width = `${(share * 100).toFixed(1)}%`
    const stage = job.stages.find((s) => s.name === 'training')
    const elapsed = stage?.startedAt ? training.at - stage.startedAt : null
    const remaining = elapsed && share > 0.02 && share < 1 ? elapsed * (1 - share) / share : null
    $('trainingText').textContent = `step ${training.step.toLocaleString()} of ${training.steps.toLocaleString()} · ` +
      `loss ${training.loss} · ${training.splats.toLocaleString()} splats` +
      (remaining ? ` · about ${duration(remaining)} to go` : '')
  }

  $('jobError').hidden = job.status !== 'failed'
  $('jobError').textContent = job.error ?? ''
  $('result').hidden = job.status !== 'succeeded'
  if (job.status === 'succeeded') renderResult(job)
}

function renderResult(job) {
  const file = (name) => `/api/jobs/${job.id}/files/${encodeURIComponent(name)}`
  const video = job.files.find((name) => name.endsWith('-turntable.mp4'))
  if (video && $('preview').dataset.src !== file(video)) {
    $('preview').src = file(video)
    $('preview').dataset.src = file(video)
    $('preview').play().catch(() => {})
  }
  const camera = job.report?.camera
  $('cameraText').textContent = camera
    ? `field of view ${camera.fovYDegrees}°, elevation ${camera.elevationDegrees}°, ` +
      `turning ${camera.direction > 0 ? '+' : '−'}, hull fit ${camera.score}`
    : ''
  $('files').replaceChildren(...job.files.map((name) =>
    element('li', {}, element('a', { href: file(name), textContent: name, download: name }))))
  const publish = $('publish')
  publish.hidden = !publish.dataset.enabled
  publish.onclick = async () => {
    publish.disabled = true
    try {
      const result = await api(`/api/jobs/${job.id}/publish`, { method: 'POST' })
      $('publishText').textContent = `Copied ${result.published.join(' and ')} into the portal.`
    } catch (error) {
      $('publishText').textContent = error.message
    } finally {
      publish.disabled = false
    }
  }
  const views = job.report?.views ?? []
  $('viewsResult').hidden = views.length === 0
  $('viewRows').replaceChildren(...views.map((view) => element('tr', {},
    element('td', { textContent: `code ${view.code}` }),
    element('td', { textContent: `${view.yaw}°` }),
    element('td', { textContent: `${view.elevation}°` }),
    element('td', { textContent: view.outlineIoU }),
    element('td', { textContent: view.used ? 'yes' : `no: ${view.reason}` }))))
}

async function follow(jobId) {
  window.clearTimeout(pollTimer)
  try {
    const job = await api(`/api/jobs/${jobId}`)
    renderJob(job)
    if (job.status === 'queued' || job.status === 'running') {
      pollTimer = window.setTimeout(() => follow(jobId), POLL_MS)
    } else {
      listJobs()
    }
  } catch (error) {
    $('job').hidden = false
    $('jobError').hidden = false
    $('jobError').textContent = error.message
  }
}

async function listJobs() {
  const jobs = await api('/api/jobs').catch(() => [])
  if (jobs.length === 0) return
  $('jobs').replaceChildren(...jobs.map((job) => element('li', {},
    element('a', { href: `#job=${job.id}`, textContent: `CAP ${job.spec.cap_id}${job.spec.paint_id ? ` / ${job.spec.paint_id}` : ''}` }),
    element('span', { className: `pill ${job.status}`, textContent: job.status }),
    element('span', { className: 'muted', textContent: `${job.spec.extra_views} views · ${job.spec.steps.toLocaleString()} steps · ${new Date(job.createdAt).toLocaleString()}` }))))
}

function fromHash() {
  const jobId = new URLSearchParams(location.hash.slice(1)).get('job')
  if (jobId) follow(jobId)
}

window.addEventListener('hashchange', fromHash)
$('capId').value = localStorage.getItem('capId') ?? ''
showGpu()
listJobs()
fromHash()
