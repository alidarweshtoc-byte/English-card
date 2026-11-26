import {useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

const cards = [
  {word: 'curious', meaning: 'فضوليّ، محبّ للاستطلاع', example: 'I’m curious about the world.'},
  {word: 'journey', meaning: 'رحلة', example: 'We began our journey before sunrise.'},
  {word: 'gentle', meaning: 'لطيف، رقيق', example: 'A gentle breeze came through the window.'},
  {word: 'discover', meaning: 'يكتشف', example: 'I like to discover new places on foot.'},
  {word: 'bright', meaning: 'ساطع، مشرق', example: 'The room is bright in the morning.'},
];

function Flashcards() {
  const [index, setIndex] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const card = cards[index];
  const canSpeak = 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;

  function move(offset: number) {
    if (canSpeak) window.speechSynthesis.cancel();
    setIndex((current) => (current + offset + cards.length) % cards.length);
    setRevealed(false);
  }

  function pronounce() {
    window.speechSynthesis.cancel();
    const speech = new SpeechSynthesisUtterance(card.word);
    speech.lang = 'en-US';
    speech.rate = 0.9;
    window.speechSynthesis.speak(speech);
  }

  return (
    <main className="layout">
      <header className="brand">
        <img src="/icon.svg" width="40" height="40" alt="" />
        <span lang="en">English Card</span>
      </header>
      <section className="intro" aria-labelledby="title">
        <p className="label">دفتر المفردات</p>
        <h1 id="title">كلمة واحدة<br />في كل مرة.</h1>
        <p>اقرأ الكلمة، وحاول تذكّر معناها، ثم اقلب البطاقة لتتأكد من إجابتك.</p>
        <small>مجموعة تدريبية · لا تحتاج إلى تسجيل الدخول</small>
      </section>
      <section className="deck" aria-label="بطاقات المفردات">
        <article className="card">
          <div className="card-top">
            <span className="label">بطاقات البداية</span>
            <span className="counter" dir="ltr" aria-label={`البطاقة ${index + 1} من ${cards.length}`}>
              {String(index + 1).padStart(2, '0')} / {String(cards.length).padStart(2, '0')}
            </span>
          </div>
          <div className="card-content" aria-live="polite">
            <h2 lang="en" dir="ltr">{card.word}</h2>
            <div id="card-answer" hidden={!revealed}>
              <p className="meaning">{card.meaning}</p>
              <p className="example" lang="en" dir="ltr">{card.example}</p>
            </div>
          </div>
          <div className="card-actions">
            <button type="button" aria-controls="card-answer" aria-expanded={revealed} onClick={() => setRevealed(!revealed)}>
              {revealed ? 'أخفِ المعنى' : 'اكشف المعنى'}
            </button>
            {canSpeak && <button type="button" className="secondary" onClick={pronounce}>استمع للنطق</button>}
          </div>
        </article>
        <nav className="card-navigation" aria-label="التنقّل بين البطاقات">
          <button type="button" className="secondary" onClick={() => move(-1)}>السابقة</button>
          <button type="button" className="secondary" onClick={() => move(1)}>التالية</button>
        </nav>
      </section>
    </main>
  );
}

const root = document.getElementById('root');
if (!root) throw new Error('Missing application root');
createRoot(root).render(<Flashcards />);
