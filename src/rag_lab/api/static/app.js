const form = document.querySelector('#search-form');
const status = document.querySelector('#status');
const results = document.querySelector('#results');
const template = document.querySelector('#result-template');
let controller;

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  controller?.abort();
  controller = new AbortController();
  const query = new FormData(form).get('query').trim();
  results.replaceChildren();
  status.textContent = '正在检索…';
  try {
    const response = await fetch('/api/v1/search', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query}), signal: controller.signal,
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message || '服务暂时不可用。');
    if (!payload.results.length) { status.textContent = '没有找到相关依据，请换一种问法。'; return; }
    status.textContent = `找到 ${payload.results.length} 条依据`;
    for (const item of payload.results) {
      const card = template.content.cloneNode(true);
      card.querySelector('.content').textContent = item.content;
      card.querySelector('.citation').textContent = `${item.citation.title} · ${item.citation.section} · 第 ${item.citation.pages} 页`;
      results.append(card);
    }
  } catch (error) {
    if (error.name !== 'AbortError') status.textContent = `${error.message} 请重试。`;
  }
});
