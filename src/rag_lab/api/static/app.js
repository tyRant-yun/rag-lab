const form = document.querySelector('#search-form');
const status = document.querySelector('#status');
const results = document.querySelector('#results');
const template = document.querySelector('#result-template');
let controller;

function fillList(selector, values) {
  const target = document.querySelector(selector);
  target.replaceChildren(...values.map((value) => {
    const item = document.createElement('li');
    item.textContent = value;
    return item;
  }));
}

async function loadKnowledgeBaseInfo() {
  try {
    const response = await fetch('/api/v1/knowledge-base');
    if (!response.ok) throw new Error();
    const info = await response.json();
    document.querySelector('#coverage').textContent = `当前覆盖：${info.coverage}`;
    fillList('#capabilities', info.capabilities);
    fillList('#guidance', info.guidance);
    fillList('#limitations', info.limitations);
    document.querySelector('#topics').textContent = `涵盖主题：${info.topics.join('、')}。`;
  } catch {
    document.querySelector('#coverage').textContent = '当前知识库范围暂时无法读取。';
  }
}

document.querySelectorAll('[data-query]').forEach((button) => {
  button.addEventListener('click', () => {
    form.elements.query.value = button.dataset.query;
    form.elements.query.focus();
  });
});

loadKnowledgeBaseInfo();

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
      card.querySelector('.citation').textContent = `教材依据：${item.citation.title} · ${item.citation.section} · 第 ${item.citation.pages} 页`;
      results.append(card);
    }
  } catch (error) {
    if (error.name !== 'AbortError') status.textContent = `${error.message} 请重试。`;
  }
});
