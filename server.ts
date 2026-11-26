import crypto from 'crypto';
import express from 'express';
import fs from 'fs';
import http from 'http';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import zlib from 'zlib';
import dotenv from 'dotenv';
import mammoth from 'mammoth';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import {
  authenticateUser,
  clearPasswordResetChallenge,
  createPasswordResetChallenge,
  createUserAccount,
  findUserByEmail,
  findUserById,
  getLeaderboard,
  getSubscribersCount,
  listAllRegisteredEmailWords,
  parseCookieHeader,
  signSessionToken,
  toPublicUser,
  updateUserPasswordByEmail,
  updateUserProfile,
  updateUserStoredData,
  validateEmail,
  validatePassword,
  verifyAndConsumePasswordResetCode,
  verifySessionToken,
  type SavedSourceFileRecord,
} from './server/accounts.ts';
import {
  isEmailDeliveryConfigured,
  sendIssueReportEmail,
  sendPasswordResetCodeEmail,
} from './server/mailer.ts';
import {
  createCommunityRoom,
  getCommunityRoomSummary,
  listPublicCommunityRooms,
  setupCommunityWebSocket,
} from './server/community.ts';
import {
  estimateWordStrengthFallback,
  listSharedWords,
  syncMissingSharedWords,
  updateSharedWordLevels,
  upsertSharedWords,
  validateSharedWordCandidate,
  type SharedWordCandidateInput,
} from './server/sharedWords.ts';
import {
  buildDualDialectInfo,
  extractReadableTextFromPdfBuffer,
  extractVocabularyFreeLocal,
  getBuiltInDictionaryEntries,
  searchBuiltInDictionary,
  translateBilingualFree,
  type BilingualDictionaryEntry,
} from './server/freeLexicon.ts';
import {
  adaptCardContextForSpecialty,
  buildDictionaryCsvString,
  deleteSavedDictionaryFile,
  getSafeDictionaryFilePath,
  getSpecialtyEntries,
  getSpecialtyMeta,
  listSavedDictionaryFiles,
  normalizeSpecialty,
  readSavedDictionaryEntries,
  saveDictionaryFileToAppPath,
} from './server/dictionaryStorage.ts';
import { alignSpokenReading } from './src/utils/readingAlign.ts';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;

function hasConfiguredGeminiKey(): boolean {
  const apiKey = (process.env.GEMINI_API_KEY || '').trim();
  return Boolean(apiKey && apiKey !== 'YOUR_API_KEY' && apiKey !== 'MY_GEMINI_API_KEY');
}

function getAiClient(apiVersion: 'v1beta' | 'v1alpha' = 'v1beta'): GoogleGenAI {
  const apiKey = (process.env.GEMINI_API_KEY || '').trim();
  if (!hasConfiguredGeminiKey()) {
    throw new Error('GEMINI_API_KEY is not configured in environment variables.');
  }
  return new GoogleGenAI({
    apiKey,
    apiVersion,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Bounded LRU-like in-memory caches to minimize free-tier API quota usage
function setBoundedCache<T>(cache: Map<string, T>, key: string, value: T, maxSize: number) {
  if (cache.size >= maxSize) {
    const oldestKey = cache.keys().next().value;
    if (oldestKey !== undefined) {
      cache.delete(oldestKey);
    }
  }
  cache.set(key, value);
}

const ttsCache = new Map<string, string>();
const MAX_TTS_CACHE = 250;
function setTtsCache(key: string, value: string) {
  setBoundedCache(ttsCache, key, value, MAX_TTS_CACHE);
}

const extractCache = new Map<
  string,
  { sourceSummary: string; preservedLayoutText?: string; words: ExtractedItem[] }
>();
const MAX_EXTRACT_CACHE = 80;

const dailyChallengeCache = new Map<string, Record<string, unknown>>();
const MAX_DAILY_CACHE = 60;

function isRetryableQuotaOrTransientError(error: unknown): boolean {
  const errObj = error as { status?: number; message?: string };
  const status = errObj?.status;
  const message = errObj?.message || (error instanceof Error ? error.message : '');
  return (
    status === 429 ||
    status === 503 ||
    status === 500 ||
    message.includes('RESOURCE_EXHAUSTED') ||
    message.includes('429') ||
    message.includes('UNAVAILABLE')
  );
}

function classifyGeminiHttpError(error: unknown): {
  status: number;
  isKeyError: boolean;
  isRateLimit: boolean;
} {
  const errObj = error as { status?: number; message?: string };
  const message = errObj?.message || (error instanceof Error ? error.message : '');
  const status = errObj?.status;

  if (
    status === 403 ||
    message.includes('API_KEY_INVALID') ||
    message.includes('PERMISSION_DENIED') ||
    message.includes('GEMINI_API_KEY')
  ) {
    return { status: 500, isKeyError: true, isRateLimit: false };
  }
  if (status === 429 || message.includes('RESOURCE_EXHAUSTED') || message.includes('429')) {
    return { status: 429, isKeyError: false, isRateLimit: true };
  }
  return { status: 500, isKeyError: false, isRateLimit: false };
}

async function withTimeout<T>(promise: Promise<T>, ms: number, label = 'operation'): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
      }),
    ]);
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer);
    }
  }
}

async function generateWithFreeFallback(
  ai: GoogleGenAI,
  params: {
    contents: unknown;
    config?: Record<string, unknown>;
  }
) {
  try {
    return await ai.models.generateContent({
      model: 'gemini-4-hydrogen',
      contents: params.contents as never,
      config: params.config as never,
    });
  } catch (primaryErr) {
    if (!isRetryableQuotaOrTransientError(primaryErr)) {
      throw primaryErr;
    }
    await new Promise((resolve) => setTimeout(resolve, 600));
    const liteConfig = params.config ? { ...params.config } : undefined;
    if (liteConfig && 'thinkingConfig' in liteConfig) {
      delete liteConfig.thinkingConfig;
    }
    try {
      return await ai.models.generateContent({
        model: 'gemini-3.1-flash-lite',
        contents: params.contents as never,
        config: liteConfig as never,
      });
    } catch {
      // Never mask the primary error (e.g. 429 RESOURCE_EXHAUSTED) with a fallback error
      throw primaryErr;
    }
  }
}

async function generateStreamWithFreeFallback(
  ai: GoogleGenAI,
  params: {
    contents: unknown;
    config?: Record<string, unknown>;
  }
) {
  try {
    return await ai.models.generateContentStream({
      model: 'gemini-4-hydrogen',
      contents: params.contents as never,
      config: params.config as never,
    });
  } catch (primaryErr) {
    if (!isRetryableQuotaOrTransientError(primaryErr)) {
      throw primaryErr;
    }
    await new Promise((resolve) => setTimeout(resolve, 600));
    const liteConfig = params.config ? { ...params.config } : undefined;
    if (liteConfig && 'thinkingConfig' in liteConfig) {
      delete liteConfig.thinkingConfig;
    }
    try {
      return await ai.models.generateContentStream({
        model: 'gemini-3.1-flash-lite',
        contents: params.contents as never,
        config: liteConfig as never,
      });
    } catch {
      throw primaryErr;
    }
  }
}

interface ExtractRequestBody {
  mode: 'text' | 'image' | 'file';
  text?: string;
  fileName?: string;
  mimeType?: string;
  base64Data?: string;
  preserveFormatting?: boolean;
  specialty?: string;
}

interface ExtractedItem {
  word: string;
  arabicMeaning: string;
  partOfSpeech: string;
  phonetic: string;
  phoneticUs?: string;
  phoneticUk?: string;
  spellingUk?: string;
  exampleEn: string;
  exampleAr: string;
  level: string;
  usageContext?: {
    synonym: string;
    collocation: string;
    collocationAr: string;
    dailySituationAr: string;
  };
}

const VALID_VOICES = new Set(['Kore', 'Puck', 'Charon', 'Fenrir', 'Zephyr']);

async function startServer() {
  const app = express();
  app.set('trust proxy', true);
  app.use(express.json({ limit: '25mb' }));

  const SESSION_COOKIE_NAME = 'mufradat_session';

  function setSessionCookie(req: express.Request, res: express.Response, token: string) {
    const isHttps =
      req.secure ||
      req.headers['x-forwarded-proto'] === 'https' ||
      req.hostname === 'localhost';
    res.cookie(SESSION_COOKIE_NAME, token, {
      httpOnly: true,
      secure: isHttps,
      sameSite: isHttps ? 'none' : 'lax',
      // Partitioned attribute for cross-site iframe preview compatibility
      ...(isHttps ? { partitioned: true } : {}),
      maxAge: 30 * 24 * 60 * 60 * 1000,
      path: '/',
    });
  }

  function clearSessionCookie(req: express.Request, res: express.Response) {
    const isHttps =
      req.secure ||
      req.headers['x-forwarded-proto'] === 'https' ||
      req.hostname === 'localhost';
    res.clearCookie(SESSION_COOKIE_NAME, {
      httpOnly: true,
      secure: isHttps,
      sameSite: isHttps ? 'none' : 'lax',
      ...(isHttps ? { partitioned: true } : {}),
      path: '/',
    });
  }

  function getAuthenticatedSession(req: express.Request) {
    const authHeader = req.headers.authorization;
    if (authHeader && authHeader.startsWith('Bearer ')) {
      const bearerToken = authHeader.slice(7).trim();
      const verified = verifySessionToken(bearerToken);
      if (verified) return { session: verified, token: bearerToken, viaCookie: false };
    }
    const cookies = parseCookieHeader(req.headers.cookie);
    const cookieToken = cookies[SESSION_COOKIE_NAME];
    if (cookieToken) {
      const verified = verifySessionToken(cookieToken);
      if (verified) return { session: verified, token: cookieToken, viaCookie: true };
    }
    return null;
  }

  function verifySameOriginOrHeader(req: express.Request): boolean {
    if (req.get('X-Requested-With') === 'XMLHttpRequest') return true;
    const origin = req.get('Origin');
    const host = req.get('X-Forwarded-Host') || req.get('Host');
    if (origin && host) {
      try {
        return new URL(origin).host === host;
      } catch {
        return false;
      }
    }
    return false;
  }

  const SERVER_BOOT_TOKEN = crypto.randomBytes(6).toString('hex');
  let cachedVersionInfo: { version: string; checkedAt: number } | null = null;

  function computeAppBuildVersion(): string {
    const now = Date.now();
    if (cachedVersionInfo && now - cachedVersionInfo.checkedAt < 2500) {
      return cachedVersionInfo.version;
    }

    const hash = crypto.createHash('sha256');
    const distIndex = path.resolve(__dirname, 'dist', 'index.html');
    const swFile = path.resolve(__dirname, 'public', 'sw.js');

    if (fs.existsSync(distIndex)) {
      hash.update(fs.readFileSync(distIndex));
      const distAssetsDir = path.resolve(__dirname, 'dist', 'assets');
      if (fs.existsSync(distAssetsDir)) {
        const assetFiles = fs.readdirSync(distAssetsDir).sort();
        hash.update(assetFiles.join(','));
      }
      if (fs.existsSync(swFile)) {
        hash.update(fs.readFileSync(swFile));
      }
    } else {
      hash.update(SERVER_BOOT_TOKEN);
      const watchFiles = [
        path.resolve(__dirname, 'package.json'),
        path.resolve(__dirname, 'index.html'),
        path.resolve(__dirname, 'src', 'App.tsx'),
        path.resolve(__dirname, 'src', 'components', 'AccountModal.tsx'),
        swFile,
      ];
      for (const filePath of watchFiles) {
        try {
          const stat = fs.statSync(filePath);
          hash.update(`${path.basename(filePath)}:${stat.size}:${Math.round(stat.mtimeMs)}`);
        } catch {
          // ignore missing optional file
        }
      }
    }

    const version = hash.digest('hex').slice(0, 16);
    cachedVersionInfo = { version, checkedAt: now };
    return version;
  }

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true });
  });

  app.get('/api/app-version', (_req, res) => {
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    res.json({
      version: computeAppBuildVersion(),
      checkedAt: Date.now(),
    });
  });

  // Always prevent browsers from caching the service worker script itself
  app.get('/sw.js', (_req, res, next) => {
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    next();
  });

  // Lightweight brute-force protection for email/password login
  const loginFailures = new Map<string, { count: number; resetAt: number }>();
  const LOGIN_WINDOW_MS = 5 * 60 * 1000;
  const MAX_LOGIN_FAILURES = 12;

  function getLoginRateKey(req: express.Request, identifier: string): string {
    const ip = req.ip || req.socket.remoteAddress || 'unknown';
    return `${ip}:${identifier.slice(0, 64)}`;
  }

  function isLoginRateLimited(key: string): boolean {
    const entry = loginFailures.get(key);
    if (!entry) return false;
    if (Date.now() > entry.resetAt) {
      loginFailures.delete(key);
      return false;
    }
    return entry.count >= MAX_LOGIN_FAILURES;
  }

  function recordLoginFailure(key: string) {
    const now = Date.now();
    const existing = loginFailures.get(key);
    if (!existing || now > existing.resetAt) {
      loginFailures.set(key, { count: 1, resetAt: now + LOGIN_WINDOW_MS });
    } else {
      existing.count += 1;
    }
  }

  function syncEmailWordsToSharedLibrary(words: unknown[], contributorName: string) {
    if (!Array.isArray(words) || words.length === 0) return;
    const validCandidates: Array<{
      word: string;
      wordLower: string;
      arabicMeaning: string;
      partOfSpeech: 'noun' | 'verb' | 'adjective' | 'adverb' | 'phrase' | 'other';
      phonetic: string;
      exampleEn: string;
      exampleAr: string;
      level: 'beginner' | 'intermediate' | 'advanced';
      deck: string;
    }> = [];
    for (const raw of words.slice(0, 500)) {
      if (!raw || typeof raw !== 'object') continue;
      const check = validateSharedWordCandidate(raw as SharedWordCandidateInput);
      if (check.valid) {
        validCandidates.push(check.cleaned);
      }
    }
    if (validCandidates.length > 0) {
      syncMissingSharedWords(validCandidates, contributorName);
    }
  }

  // Email + Easy Password Registration
  app.post('/api/auth/register', (req, res) => {
    try {
      const { email, password, initialData } = (req.body || {}) as {
        email?: string;
        password?: string;
        initialData?: {
          words?: unknown[];
          savedFiles?: SavedSourceFileRecord[];
          selectedVoice?: string;
          dailyProgress?: { completedDates: string[]; streak: number; totalCompleted: number };
        };
      };

      const result = createUserAccount({
        email: String(email ?? ''),
        password: String(password ?? ''),
        initialData: initialData
          ? {
              words: Array.isArray(initialData.words) ? initialData.words : [],
              savedFiles: Array.isArray(initialData.savedFiles) ? initialData.savedFiles : [],
              selectedVoice: initialData.selectedVoice || 'Kore',
              dailyProgress: initialData.dailyProgress ?? {
                completedDates: [],
                streak: 0,
                totalCompleted: 0,
              },
            }
          : undefined,
      });

      if (!result.ok) {
        const fail = result as { ok: false; status: number; error: string };
        res.status(fail.status).json({ error: fail.error });
        return;
      }

      syncEmailWordsToSharedLibrary(result.user.data.words, result.user.username);

      const token = signSessionToken(result.user);
      setSessionCookie(req, res, token);

      res.status(201).json({
        user: toPublicUser(result.user),
        token,
        data: result.user.data,
        subscribersCount: getSubscribersCount(),
      });
    } catch (error) {
      console.error('Error in /api/auth/register:', error);
      res.status(500).json({ error: 'تعذر إنشاء الحساب حالياً. حاول مجدداً.' });
    }
  });

  // Email + Password Login (supports legacy username accounts seamlessly)
  app.post('/api/auth/login', (req, res) => {
    try {
      const { email, username, password } = (req.body || {}) as {
        email?: string;
        username?: string;
        password?: string;
      };

      const identifier = String(email ?? username ?? '').trim().toLowerCase();
      const rateKey = getLoginRateKey(req, identifier);
      if (isLoginRateLimited(rateKey)) {
        res.status(429).json({
          error: 'محاولات دخول متكررة. يرجى الانتظار بضع دقائق ثم المحاولة مجدداً.',
        });
        return;
      }

      const result = authenticateUser({
        email: typeof email === 'string' ? email : undefined,
        username: typeof username === 'string' ? username : undefined,
        password: String(password ?? ''),
      });

      if (!result.ok) {
        recordLoginFailure(rateKey);
        const fail = result as { ok: false; status: number; error: string };
        res.status(fail.status).json({ error: fail.error });
        return;
      }

      loginFailures.delete(rateKey);
      const token = signSessionToken(result.user);
      setSessionCookie(req, res, token);

      res.json({
        user: toPublicUser(result.user),
        token,
        data: result.user.data,
        subscribersCount: getSubscribersCount(),
      });
    } catch (error) {
      console.error('Error in /api/auth/login:', error);
      res.status(500).json({ error: 'تعذر تسجيل الدخول حالياً.' });
    }
  });

  // Current Authenticated Session
  app.get('/api/auth/me', (req, res) => {
    const auth = getAuthenticatedSession(req);
    if (!auth) {
      res.json({ user: null });
      return;
    }
    const user = findUserById(auth.session.uid);
    if (!user) {
      clearSessionCookie(req, res);
      res.json({ user: null });
      return;
    }
    res.json({
      user: toPublicUser(user),
      token: auth.token,
      data: user.data,
      subscribersCount: getSubscribersCount(),
    });
  });

  // Logout
  app.post('/api/auth/logout', (req, res) => {
    if (!verifySameOriginOrHeader(req)) {
      res.status(403).json({ error: 'طلب غير مصرح به.' });
      return;
    }
    clearSessionCookie(req, res);
    res.json({ ok: true });
  });

  // Forgot Password: Send 6-digit verification code to registered email
  const forgotPasswordIpLimiter = new Map<string, { count: number; resetAt: number }>();

  app.post('/api/auth/forgot-password', async (req, res) => {
    try {
      if (!verifySameOriginOrHeader(req)) {
        res.status(403).json({ error: 'طلب غير مصرح به.' });
        return;
      }

      const ip = req.ip || req.socket.remoteAddress || 'unknown';
      const now = Date.now();
      const ipEntry = forgotPasswordIpLimiter.get(ip);
      if (ipEntry && now < ipEntry.resetAt && ipEntry.count >= 12) {
        res.status(429).json({
          error: 'تم إرسال طلبات كثيرة من هذا الجهاز. يرجى الانتظار بضع دقائق.',
        });
        return;
      }
      if (!ipEntry || now >= ipEntry.resetAt) {
        forgotPasswordIpLimiter.set(ip, { count: 1, resetAt: now + 10 * 60 * 1000 });
      } else {
        ipEntry.count += 1;
      }

      const { email } = (req.body || {}) as { email?: string };
      const eCheck = validateEmail(String(email ?? ''));
      if (!eCheck.valid) {
        res.status(400).json({ error: eCheck.error || 'يرجى إدخال بريد إلكتروني صحيح.' });
        return;
      }

      if (!isEmailDeliveryConfigured()) {
        res.status(503).json({
          error:
            'خدمة إرسال أكواد التفعيل غير مهيأة حالياً على الخادم. يرجى إضافة بيانات البريد (RESEND_API_KEY أو إعدادات SMTP) في متغيرات البيئة.',
        });
        return;
      }

      const user = findUserByEmail(eCheck.emailLower);
      if (user && user.email) {
        const challenge = createPasswordResetChallenge(eCheck.emailLower);
        if (!challenge.ok) {
          const fail = challenge as {
            ok: false;
            status: number;
            retryAfterSeconds?: number;
            error: string;
          };
          res.status(fail.status).json({
            error: fail.error,
            retryAfterSeconds: fail.retryAfterSeconds,
          });
          return;
        }

        try {
          await sendPasswordResetCodeEmail({
            toEmail: user.email,
            username: user.username,
            code: challenge.code,
            expiresMinutes: challenge.expiresMinutes,
          });
        } catch (sendErr) {
          clearPasswordResetChallenge(eCheck.emailLower);
          console.error('Failed to send password reset code:', sendErr);
          res.status(500).json({
            error: 'تعذر إرسال كود التفعيل إلى البريد الإلكتروني حالياً. يرجى المحاولة بعد قليل.',
          });
          return;
        }
      }

      res.json({
        ok: true,
        expiresMinutes: 10,
        message:
          'إذا كان هذا البريد مسجلاً في التطبيق، فقد تم إرسال كود التفعيل المكون من 6 أرقام إليه. افحص صندوق الوارد والرسائل غير المرغوب فيها.',
      });
    } catch (error) {
      console.error('Error in /api/auth/forgot-password:', error);
      res.status(500).json({ error: 'تعذر معالجة طلب استرجاع كلمة المرور حالياً.' });
    }
  });

  // Reset Password: Verify 6-digit email code, update password, preserve all email data, and sign in
  app.post('/api/auth/reset-password', (req, res) => {
    try {
      if (!verifySameOriginOrHeader(req)) {
        res.status(403).json({ error: 'طلب غير مصرح به.' });
        return;
      }

      const { email, code, newPassword } = (req.body || {}) as {
        email?: string;
        code?: string;
        newPassword?: string;
      };

      const eCheck = validateEmail(String(email ?? ''));
      if (!eCheck.valid) {
        res.status(400).json({ error: eCheck.error || 'يرجى إدخال بريد إلكتروني صحيح.' });
        return;
      }

      const pCheck = validatePassword(String(newPassword ?? ''));
      if (!pCheck.valid) {
        res.status(400).json({ error: pCheck.error || 'كلمة المرور غير صالحة.' });
        return;
      }

      const codeCheck = verifyAndConsumePasswordResetCode({
        email: eCheck.emailLower,
        code: String(code ?? ''),
      });
      if (!codeCheck.ok) {
        const fail = codeCheck as { ok: false; status: number; error: string };
        res.status(fail.status).json({ error: fail.error });
        return;
      }

      const updated = updateUserPasswordByEmail(eCheck.emailLower, String(newPassword ?? ''));
      if (!updated.ok) {
        const fail = updated as { ok: false; status: number; error: string };
        res.status(fail.status).json({ error: fail.error });
        return;
      }

      const token = signSessionToken(updated.user);
      setSessionCookie(req, res, token);

      res.json({
        user: toPublicUser(updated.user),
        token,
        data: updated.user.data,
        subscribersCount: getSubscribersCount(),
      });
    } catch (error) {
      console.error('Error in /api/auth/reset-password:', error);
      res.status(500).json({ error: 'تعذر تحديث كلمة المرور حالياً.' });
    }
  });

  const ISSUE_REPORT_RECIPIENT = (
    process.env.ISSUE_REPORT_EMAIL || 'engalibaba313@gmail.com'
  ).trim();
  const ISSUE_REPORTS_FILE = path.resolve(process.cwd(), 'data', 'issue-reports.json');
  const issueReportIpLimiter = new Map<string, { count: number; resetAt: number }>();

  function persistIssueReportRecord(record: {
    id: string;
    category: string;
    message: string;
    reporterName: string;
    reporterEmail?: string;
    userAgent?: string;
    deliveredViaEmail: boolean;
    createdAt: number;
  }) {
    try {
      const dir = path.dirname(ISSUE_REPORTS_FILE);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      let existing: unknown[] = [];
      if (fs.existsSync(ISSUE_REPORTS_FILE)) {
        try {
          const parsed = JSON.parse(fs.readFileSync(ISSUE_REPORTS_FILE, 'utf-8'));
          if (Array.isArray(parsed)) existing = parsed;
        } catch {
          existing = [];
        }
      }
      const next = [record, ...existing].slice(0, 500);
      fs.writeFileSync(ISSUE_REPORTS_FILE, JSON.stringify(next, null, 2), 'utf-8');
    } catch (err) {
      console.error('Failed to persist issue report backup:', err);
    }
  }

  // User Problem / Feedback Report Endpoint -> delivers to engalibaba313@gmail.com
  app.post('/api/report-issue', async (req, res) => {
    try {
      if (!verifySameOriginOrHeader(req)) {
        res.status(403).json({ error: 'طلب غير مصرح به.' });
        return;
      }

      const ip = (req.ip || req.socket.remoteAddress || 'unknown').slice(0, 64);
      const now = Date.now();
      const entry = issueReportIpLimiter.get(ip);
      if (entry && now < entry.resetAt && entry.count >= 8) {
        res.status(429).json({
          error: 'تم إرسال عدة بلاغات مؤخراً. يرجى الانتظار بضع دقائق قبل الإرسال مجدداً.',
        });
        return;
      }
      if (!entry || now >= entry.resetAt) {
        issueReportIpLimiter.set(ip, { count: 1, resetAt: now + 15 * 60 * 1000 });
      } else {
        entry.count += 1;
      }

      const body = (req.body || {}) as {
        category?: string;
        message?: string;
        reporterEmail?: string;
        reporterName?: string;
      };

      const cleanMessage = String(body.message ?? '').trim().slice(0, 2500);
      if (cleanMessage.length < 5) {
        res.status(400).json({ error: 'يرجى كتابة تفاصيل المشكلة أو الملاحظة (5 أحرف على الأقل).' });
        return;
      }

      const cleanCategory =
        String(body.category ?? 'مشكلة تقنية أو خطأ').trim().slice(0, 80) || 'مشكلة تقنية أو خطأ';

      const auth = getAuthenticatedSession(req);
      const authUser = auth ? findUserById(auth.session.uid) : null;

      let cleanReporterEmail: string | undefined;
      const rawEmail = String(body.reporterEmail ?? '').trim();
      if (rawEmail) {
        const eCheck = validateEmail(rawEmail);
        if (!eCheck.valid) {
          res.status(400).json({ error: eCheck.error || 'البريد الإلكتروني المدخل غير صالح.' });
          return;
        }
        cleanReporterEmail = eCheck.emailLower;
      } else if (authUser?.email) {
        cleanReporterEmail = authUser.email;
      }

      const cleanReporterName =
        String(body.reporterName ?? '').trim().slice(0, 60) ||
        authUser?.username ||
        'مستخدم التطبيق';

      const userAgent = String(req.headers['user-agent'] || '').slice(0, 220);

      let deliveredViaEmail = false;
      if (isEmailDeliveryConfigured()) {
        try {
          await sendIssueReportEmail({
            toEmail: ISSUE_REPORT_RECIPIENT,
            category: cleanCategory,
            message: cleanMessage,
            reporterName: cleanReporterName,
            reporterEmail: cleanReporterEmail,
            userAgent,
          });
          deliveredViaEmail = true;
        } catch (emailErr) {
          console.error('Issue report email delivery failed, falling back to local backup:', emailErr);
        }
      }

      persistIssueReportRecord({
        id: `rep-${now}-${crypto.randomBytes(3).toString('hex')}`,
        category: cleanCategory,
        message: cleanMessage,
        reporterName: cleanReporterName,
        reporterEmail: cleanReporterEmail,
        userAgent,
        deliveredViaEmail,
        createdAt: now,
      });

      const mailSubject = encodeURIComponent(`[English Card] ${cleanCategory}`);
      const mailBody = encodeURIComponent(
        `المرسل: ${cleanReporterName}${cleanReporterEmail ? ` (${cleanReporterEmail})` : ''}\nالتصنيف: ${cleanCategory}\n\nالملاحظة:\n${cleanMessage}`
      );
      const mailtoFallbackUrl = `mailto:${ISSUE_REPORT_RECIPIENT}?subject=${mailSubject}&body=${mailBody}`;

      res.json({
        ok: true,
        deliveredVia: deliveredViaEmail ? 'email' : 'saved_fallback',
        mailtoFallbackUrl,
        message: deliveredViaEmail
          ? `تم إرسال ملاحظتك بنجاح إلى البريد الإلكتروني (${ISSUE_REPORT_RECIPIENT}). شكراً لمساهمتك!`
          : `تم تسجيل بلاغك وحفظه بنجاح، ويمكنك إرساله فوراً بالبريد إلى (${ISSUE_REPORT_RECIPIENT}) عبر الزر أدناه.`,
      });
    } catch (error) {
      console.error('Error in POST /api/report-issue:', error);
      res.status(500).json({ error: 'تعذر إرسال الملاحظة حالياً. حاول مجدداً.' });
    }
  });

  // Sync Authenticated User Data (Words, Saved Files, Voice, Daily Streak)
  app.put('/api/user/data', (req, res) => {
    const auth = getAuthenticatedSession(req);
    if (!auth) {
      res.status(401).json({ error: 'يرجى تسجيل الدخول لحفظ التغييرات في حسابك.' });
      return;
    }
    if (auth.viaCookie && !verifySameOriginOrHeader(req)) {
      res.status(403).json({ error: 'طلب غير مصرح به.' });
      return;
    }

    const body = (req.body || {}) as {
      words?: unknown[];
      savedFiles?: SavedSourceFileRecord[];
      selectedVoice?: string;
      dailyProgress?: { completedDates: string[]; streak: number; totalCompleted: number };
    };

    const wordsList = Array.isArray(body.words) ? body.words : [];
    const ok = updateUserStoredData(auth.session.uid, {
      words: wordsList,
      savedFiles: Array.isArray(body.savedFiles) ? body.savedFiles : [],
      selectedVoice: typeof body.selectedVoice === 'string' ? body.selectedVoice : 'Kore',
      dailyProgress: body.dailyProgress ?? {
        completedDates: [],
        streak: 0,
        totalCompleted: 0,
      },
    });

    if (!ok) {
      res.status(404).json({ error: 'الحساب غير موجود.' });
      return;
    }

    syncEmailWordsToSharedLibrary(wordsList, auth.session.username || 'مستخدم');
    res.json({ ok: true });
  });

  // Update Authenticated User Profile (Display Name & Avatar)
  app.put('/api/user/profile', (req, res) => {
    const auth = getAuthenticatedSession(req);
    if (!auth) {
      res.status(401).json({ error: 'يرجى تسجيل الدخول لتعديل بيانات الحساب.' });
      return;
    }
    if (auth.viaCookie && !verifySameOriginOrHeader(req)) {
      res.status(403).json({ error: 'طلب غير مصرح به.' });
      return;
    }

    const body = (req.body || {}) as {
      username?: unknown;
      avatarDataUrl?: unknown;
    };

    const result = updateUserProfile(auth.session.uid, {
      username: body.username,
      avatarDataUrl: body.avatarDataUrl,
    });

    if (!result.ok) {
      const fail = result as { ok: false; status: number; error: string };
      res.status(fail.status).json({ error: fail.error });
      return;
    }

    const freshToken = signSessionToken(result.user);
    setSessionCookie(req, res, freshToken);

    res.json({
      user: toPublicUser(result.user),
      token: freshToken,
    });
  });

  // Public Study & Level Leaderboard (Zero emails exposed)
  app.get('/api/leaderboard', (_req, res) => {
    try {
      const entries = getLeaderboard();
      res.json({
        entries,
        totalUsers: entries.length,
      });
    } catch (error) {
      console.error('Error in GET /api/leaderboard:', error);
      res.status(500).json({ error: 'تعذر تحميل قائمة المتصدرين حالياً.' });
    }
  });

      type ExtractPart =
        | { text: string }
        | { inlineData: { mimeType: string; data: string } }
        | { fileData: { fileUri: string; mimeType: string } };

      const EXTRACT_INSTRUCTION_PROMPT = `You are an expert bilingual English-Arabic lexicographer and language teacher.
Analyze the provided input (text, document, or image) and extract the most useful English vocabulary words or key phrasal expressions for an Arabic-speaking learner.

Rules:
1. Extract up to 25 distinct, meaningful English words or short idiomatic phrases from the input. Skip trivial stopwords like "the", "a", "an", "is", "in", "of" unless they are part of a meaningful phrase or the user specifically listed very basic single words.
2. If the user pasted a direct list of words (even basic ones), include ALL of them (up to 35).
3. Normalize each word to clean base/dictionary form where appropriate (keep specific forms if clearly intended as vocabulary items) with clean capitalization (lowercase for common words, capitalized for proper nouns).
4. Provide:
   - word: The English word or short expression.
   - arabicMeaning: Clear, accurate, natural Arabic translation (include 1-2 core meanings separated by " / " if helpful).
   - partOfSpeech: One of "noun", "verb", "adjective", "adverb", "phrase", "other".
   - phonetic: Accurate IPA pronunciation enclosed in slashes, e.g., "/ˈel.ə.kwənt/".
   - exampleEn: A clear, natural, modern English sentence demonstrating the word in context.
   - exampleAr: Accurate Arabic translation of exampleEn.
   - level: Classify strictly by the lexical strength and CEFR difficulty of the word itself:
     * "beginner": High-frequency everyday basic words (A1-A2, e.g., book, water, happy, walk, buy, friend).
     * "intermediate": Practical workplace, study, media, and moderately nuanced words (B1-B2, e.g., collaborate, efficient, deadline, strategy, opportunity).
     * "advanced": Strong academic, literary, formal, technical, or low-frequency expressive words/idioms (C1-C2, e.g., resilient, paradigm, ubiquitous, articulate, mitigate, unprecedented).
5. Also return a short 1-sentence Arabic summary of the extracted source in "sourceSummary".`;

      async function executeVocabExtraction(ai: GoogleGenAI, parts: ExtractPart[]) {
        const response = await generateWithFreeFallback(ai, {
          contents: [{ role: 'user', parts }],
          config: {
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                sourceSummary: {
                  type: Type.STRING,
                  description: 'A short Arabic summary describing what was extracted.',
                },
                preservedLayoutText: {
                  type: Type.STRING,
                  description:
                    'Optional formatted excerpt of headings, tables, and lists preserving the uploaded document layout.',
                },
                words: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: {
                      word: { type: Type.STRING },
                      arabicMeaning: { type: Type.STRING },
                      partOfSpeech: { type: Type.STRING },
                      phonetic: { type: Type.STRING },
                      exampleEn: { type: Type.STRING },
                      exampleAr: { type: Type.STRING },
                      level: { type: Type.STRING },
                    },
                    required: [
                      'word',
                      'arabicMeaning',
                      'partOfSpeech',
                      'phonetic',
                      'exampleEn',
                      'exampleAr',
                      'level',
                    ],
                  },
                },
              },
              required: ['sourceSummary', 'words'],
            },
          },
        });

        const rawText = (response.text ?? '{}')
          .trim()
          .replace(/^```json\s*/i, '')
          .replace(/^```\s*/i, '')
          .replace(/\s*```$/i, '');
        const parsed = JSON.parse(rawText) as {
          sourceSummary?: string;
          preservedLayoutText?: string;
          words?: ExtractedItem[];
        };

        const seen = new Set<string>();
        const cleanedWords: ExtractedItem[] = [];
        for (const item of parsed.words ?? []) {
          const normalized = (item.word ?? '').trim();
          if (!normalized) continue;
          const lower = normalized.toLowerCase();
          if (seen.has(lower)) continue;
          seen.add(lower);

          const pos = ['noun', 'verb', 'adjective', 'adverb', 'phrase', 'other'].includes(
            (item.partOfSpeech ?? '').toLowerCase()
          )
            ? (item.partOfSpeech ?? 'other').toLowerCase()
            : 'other';

          const level = ['beginner', 'intermediate', 'advanced'].includes(
            (item.level ?? '').toLowerCase()
          )
            ? (item.level ?? 'intermediate').toLowerCase()
            : 'intermediate';

          const cleanPhonetic = (item.phonetic ?? '').trim() || `/${lower}/`;
          const dualDialect = buildDualDialectInfo(normalized, cleanPhonetic);
          const cleanArabic = (item.arabicMeaning ?? '').trim();

          cleanedWords.push({
            word: normalized,
            arabicMeaning: cleanArabic,
            partOfSpeech: pos,
            phonetic: dualDialect.phoneticUs,
            phoneticUs: dualDialect.phoneticUs,
            phoneticUk: dualDialect.phoneticUk,
            spellingUk: dualDialect.spellingUk,
            exampleEn: (item.exampleEn ?? '').trim(),
            exampleAr: (item.exampleAr ?? '').trim(),
            level,
          });
        }

        return { parsed, cleanedWords };
      }

      function handleExtractionError(res: express.Response, error: unknown, routeLabel: string) {
        console.error(`Error in ${routeLabel}:`, error);
        const errObj = error as { status?: number; message?: string };
        const message = errObj?.message || (error instanceof Error ? error.message : '');
        const status = errObj?.status;

        if (status === 429 || message.includes('RESOURCE_EXHAUSTED') || message.includes('429')) {
          res.status(429).json({
            error: 'الخادم مزدحم مؤقتاً. يمكنك لصق النص مباشرة للاستخراج الفوري عبر القاموس المدمج المجاني.',
          });
          return;
        }
        res.status(500).json({
          error: 'تعذر استخراج الكلمات من هذا الملف حالياً. يمكنك لصق النص مباشرة أو رفع ملف نصي/PDF نصي للاستخراج المجاني فوراً.',
        });
      }

      const CHUNK_UPLOAD_DIR = path.resolve(os.tmpdir(), 'mufradat-uploads');
      const MAX_UPLOAD_BYTES = 65 * 1024 * 1024;
      const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

      function ensureChunkUploadDir() {
        if (!fs.existsSync(CHUNK_UPLOAD_DIR)) {
          fs.mkdirSync(CHUNK_UPLOAD_DIR, { recursive: true });
        }
      }

      function getSafeChunkPath(uploadId: string): string | null {
        if (!UUID_RE.test(uploadId)) return null;
        ensureChunkUploadDir();
        const target = path.resolve(CHUNK_UPLOAD_DIR, `${uploadId}.bin`);
        if (!target.startsWith(CHUNK_UPLOAD_DIR + path.sep)) return null;
        return target;
      }

      // Periodically remove stale temporary chunk uploads (> 10 mins)
      setInterval(() => {
        try {
          if (!fs.existsSync(CHUNK_UPLOAD_DIR)) return;
          const cutoff = Date.now() - 10 * 60 * 1000;
          for (const entry of fs.readdirSync(CHUNK_UPLOAD_DIR)) {
            const full = path.resolve(CHUNK_UPLOAD_DIR, entry);
            if (!full.startsWith(CHUNK_UPLOAD_DIR + path.sep)) continue;
            const stat = fs.statSync(full);
            if (stat.mtimeMs < cutoff) {
              fs.rmSync(full, { force: true });
            }
          }
        } catch {
          // ignore cleanup failure
        }
      }, 5 * 60 * 1000).unref();

      // Chunk receiver for large compressed files (> 10MB, up to 65MB)
      app.post(
        '/api/extract-vocab-chunk',
        express.raw({ type: 'application/octet-stream', limit: '8mb' }),
        (req, res) => {
          try {
            const rawId = String(req.headers['x-upload-id'] || '').trim();
            const uploadId = rawId || crypto.randomUUID();
            const chunkPath = getSafeChunkPath(uploadId);
            if (!chunkPath) {
              res.status(400).json({ error: 'معرف الرفع غير صالح.' });
              return;
            }
            const chunkBuf = Buffer.isBuffer(req.body) ? req.body : Buffer.alloc(0);
            if (chunkBuf.length === 0) {
              res.status(400).json({ error: 'الجزء المرفوع فارغ.' });
              return;
            }

            const existingSize = fs.existsSync(chunkPath) ? fs.statSync(chunkPath).size : 0;
            if (existingSize + chunkBuf.length > MAX_UPLOAD_BYTES) {
              fs.rmSync(chunkPath, { force: true });
              res.status(413).json({ error: 'حجم الملف يتجاوز الحد الأقصى المسموح (65 ميجابايت).' });
              return;
            }

            fs.appendFileSync(chunkPath, chunkBuf);
            res.json({ ok: true, uploadId, receivedBytes: existingSize + chunkBuf.length });
          } catch (error) {
            console.error('Error in /api/extract-vocab-chunk:', error);
            res.status(500).json({ error: 'تعذر استلام جزء الملف.' });
          }
        }
      );

      // Binary / chunked / gzipped file extraction (supports up to 60MB files)
      app.post(
        '/api/extract-vocab-file',
        express.raw({ type: 'application/octet-stream', limit: '15mb' }),
        async (req, res) => {
          let tempChunkPath: string | null = null;
          let tempPdfPath: string | null = null;
          let uploadedGeminiFileName: string | null = null;
          let aiClient: GoogleGenAI | null = null;

          try {
            const rawUploadId = String(req.headers['x-upload-id'] || '').trim();
            const rawFileName = String(req.headers['x-file-name'] || 'document');
            let fileName = 'document';
            try {
              fileName = decodeURIComponent(rawFileName).slice(0, 180);
            } catch {
              fileName = rawFileName.slice(0, 180);
            }
            const lowerName = fileName.toLowerCase();
            const mimeType = String(req.headers['x-mime-type'] || 'application/octet-stream').slice(
              0,
              120
            );
            const isGzipped = String(req.headers['x-compressed'] || '').toLowerCase() === 'gzip';
            const keepLayout = String(req.headers['x-preserve-formatting'] || 'true') !== 'false';

            let fileBuffer: Buffer;
            if (rawUploadId) {
              const safePath = getSafeChunkPath(rawUploadId);
              if (!safePath || !fs.existsSync(safePath)) {
                res.status(400).json({ error: 'انتهت جلسة رفع الملف. يرجى إعادة المحاولة.' });
                return;
              }
              tempChunkPath = safePath;
              fileBuffer = fs.readFileSync(safePath);
            } else {
              fileBuffer = Buffer.isBuffer(req.body) ? req.body : Buffer.alloc(0);
            }

            if (fileBuffer.length === 0) {
              res.status(400).json({ error: 'لم يتم استلام محتوى الملف.' });
              return;
            }

            if (isGzipped) {
              try {
                fileBuffer = zlib.gunzipSync(fileBuffer, { maxOutputLength: MAX_UPLOAD_BYTES });
              } catch {
                res.status(400).json({ error: 'تعذر فك ضغط الملف المرفوع أو أنه يتجاوز الحد المسموح.' });
                return;
              }
            }

            if (fileBuffer.length > MAX_UPLOAD_BYTES) {
              res.status(413).json({ error: 'حجم الملف يتجاوز الحد الأقصى المسموح (65 ميجابايت).' });
              return;
            }

            const parts: ExtractPart[] = [];
            let fallbackReadableText: string | null = null;
            const useGemini = hasConfiguredGeminiKey();

            if (
              lowerName.endsWith('.docx') ||
              mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ) {
              const result = await mammoth.extractRawText({ buffer: fileBuffer });
              const extractedDocText = (result.value ?? '').trim();
              if (!extractedDocText) {
                res.status(400).json({ error: 'لم يتم العثور على نص مقروء داخل ملف Word.' });
                return;
              }
              fallbackReadableText = extractedDocText;
              parts.push({
                text: `${EXTRACT_INSTRUCTION_PROMPT}\n\n${
                  keepLayout
                    ? 'Preserve the original section headings, lists, and reading order from the DOCX content, and also include a concise "preservedLayoutText" summary preserving headings/lists.'
                    : ''
                }\nExtracted text from DOCX file (${fileName}):\n"""\n${extractedDocText.slice(0, 30000)}\n"""`,
              });
            } else if (mimeType === 'application/pdf' || lowerName.endsWith('.pdf')) {
              if (fileBuffer.subarray(0, 5).toString('ascii') !== '%PDF-') {
                res.status(400).json({ error: 'الملف المرفوع لا يبدو كملف PDF صالح.' });
                return;
              }

              fallbackReadableText = extractReadableTextFromPdfBuffer(fileBuffer);

              if (useGemini) {
                aiClient = getAiClient();
                const pdfPrompt = `${EXTRACT_INSTRUCTION_PROMPT}

PDF Formatting & Layout Rules:
- Read all pages of the attached PDF carefully${keepLayout ? ' while strictly preserving the document structure (headings, numbered lists, multi-column tables, word-to-meaning rows, and original sequence)' : ''}.
- If the PDF contains an English/Arabic vocabulary table or glossary, preserve each exact English term paired with its listed Arabic translation and context sentence in the original order.
- Also provide "preservedLayoutText": a clean, readable plain-text/Markdown reconstruction (up to 2000 characters) of the key PDF sections, headings, and lists preserving the original formatting.`;

                if (fileBuffer.length <= 14 * 1024 * 1024) {
                  parts.push({
                    inlineData: {
                      mimeType: 'application/pdf',
                      data: fileBuffer.toString('base64'),
                    },
                  });
                } else {
                  // Large PDFs (>14MB) use Gemini Files API cleanly
                  ensureChunkUploadDir();
                  tempPdfPath = path.resolve(CHUNK_UPLOAD_DIR, `${crypto.randomUUID()}.pdf`);
                  fs.writeFileSync(tempPdfPath, fileBuffer);
                  let uploaded = await aiClient.files.upload({
                    file: tempPdfPath,
                    config: { mimeType: 'application/pdf', displayName: fileName },
                  });
                  uploadedGeminiFileName = uploaded.name || null;

                  let attempts = 0;
                  while (uploaded.state === 'PROCESSING' && uploaded.name && attempts < 15) {
                    await new Promise((r) => setTimeout(r, 1000));
                    uploaded = await aiClient.files.get({ name: uploaded.name });
                    attempts += 1;
                  }

                  if (uploaded.uri && uploaded.state !== 'FAILED') {
                    parts.push({
                      fileData: {
                        fileUri: uploaded.uri,
                        mimeType: uploaded.mimeType || 'application/pdf',
                      },
                    });
                  }
                }
                if (parts.length > 0) {
                  parts.push({ text: pdfPrompt });
                }
              }
            } else if (mimeType.startsWith('image/')) {
              parts.push({
                inlineData: {
                  mimeType,
                  data: fileBuffer.toString('base64'),
                },
              });
              parts.push({
                text: `${EXTRACT_INSTRUCTION_PROMPT}\n\nExtract the English vocabulary words from this image.`,
              });
            } else {
              const decodedText = fileBuffer.toString('utf-8').trim();
              if (!decodedText) {
                res.status(400).json({ error: 'الملف فارغ أو لا يحتوي على نص مقروء.' });
                return;
              }
              fallbackReadableText = decodedText;
              parts.push({
                text: `${EXTRACT_INSTRUCTION_PROMPT}\n\n${
                  keepLayout
                    ? 'Preserve headings, columns, and line structure from this file and include a concise formatted preview in "preservedLayoutText".'
                    : ''
                }\nContent from uploaded file (${fileName}):\n"""\n${decodedText.slice(0, 30000)}\n"""`,
              });
            }

            if (useGemini && parts.length > 0) {
              try {
                if (!aiClient) aiClient = getAiClient();
                const { parsed, cleanedWords } = await withTimeout(
                  executeVocabExtraction(aiClient, parts),
                  fallbackReadableText ? 8000 : 25000,
                  'extract-vocab-file'
                );
                if (cleanedWords.length > 0) {
                  res.json({
                    sourceSummary:
                      parsed.sourceSummary || `تم استخراج ${cleanedWords.length} كلمة بنجاح.`,
                    preservedLayoutText:
                      typeof parsed.preservedLayoutText === 'string' &&
                      parsed.preservedLayoutText.trim()
                        ? parsed.preservedLayoutText.trim().slice(0, 3000)
                        : undefined,
                    words: cleanedWords,
                  });
                  return;
                }
              } catch (geminiErr) {
                if (!fallbackReadableText || fallbackReadableText.length < 3) {
                  throw geminiErr;
                }
              }
            }

            if (fallbackReadableText && fallbackReadableText.length >= 2) {
              const freeResult = await extractVocabularyFreeLocal(fallbackReadableText);
              if (freeResult.words.length === 0) {
                res.status(422).json({
                  error: 'لم يتم العثور على كلمات إنجليزية واضحة في الملف المدخل. جرّب ملفاً أوضح.',
                });
                return;
              }
              res.json({
                ...freeResult,
                extractionMode: 'free-dictionary',
              });
              return;
            }

            res.status(422).json({
              error:
                'هذا الملف عبارة عن صورة أو PDF مصوّر لا يحتوي على نص قابل للنسخ. يرجى لصق النص مباشرة أو رفع ملف نصي/Word للاستخراج المجاني الفوري.',
            });
          } catch (error) {
            handleExtractionError(res, error, '/api/extract-vocab-file');
          } finally {
            if (tempChunkPath && fs.existsSync(tempChunkPath)) {
              fs.rmSync(tempChunkPath, { force: true });
            }
            if (tempPdfPath && fs.existsSync(tempPdfPath)) {
              fs.rmSync(tempPdfPath, { force: true });
            }
            if (uploadedGeminiFileName && aiClient) {
              void aiClient.files.delete({ name: uploadedGeminiFileName }).catch(() => {
                // ignore remote cleanup error
              });
            }
          }
        }
      );

      app.post('/api/extract-vocab', async (req, res) => {
        try {
          const body = req.body as ExtractRequestBody;

          if (!body || !body.mode) {
            res.status(400).json({ error: 'يرجى تحديد طريقة الإدخال بشكل صحيح.' });
            return;
          }

          const parts: ExtractPart[] = [];
          let fallbackReadableText: string | null = null;
          let textCacheKey: string | null = null;
          const useGemini = hasConfiguredGeminiKey();

          if (body.mode === 'text') {
            const cleanText = (body.text ?? '').trim();
            if (!cleanText) {
              res.status(400).json({ error: 'يرجى كتابة أو لصق نص أو كلمات إنجليزية أولاً.' });
              return;
            }
            const specialty = normalizeSpecialty(body.specialty);
            const specialtyMeta = getSpecialtyMeta(specialty);
            fallbackReadableText = cleanText;
            if (cleanText.length <= 4000) {
              textCacheKey = crypto
                .createHash('sha256')
                .update(`${specialty}::${cleanText.toLowerCase()}`)
                .digest('hex');
              const cached = extractCache.get(textCacheKey);
              if (cached) {
                res.json(cached);
                return;
              }
            }
            const specialtyHint =
              specialty !== 'general'
                ? `\nProfessional Specialty Context: ${specialtyMeta.englishDomain} (${specialtyMeta.label}). Tailor every "exampleEn" sentence, "exampleAr" translation, and "arabicMeaning" nuance specifically to this professional domain.`
                : '';
            parts.push({
              text: `${EXTRACT_INSTRUCTION_PROMPT}${specialtyHint}\n\nUser input text:\n"""\n${cleanText.slice(0, 20000)}\n"""`,
            });
          } else if (body.mode === 'image') {
            if (!body.base64Data || !body.mimeType) {
              res.status(400).json({ error: 'بيانات الصورة غير مكتملة. يرجى رفع صورة صالحة.' });
              return;
            }
            if (!body.mimeType.startsWith('image/')) {
              res.status(400).json({ error: 'الملف المرفوع ليس صورة صالحة.' });
              return;
            }
            parts.push({
              inlineData: {
                mimeType: body.mimeType,
                data: body.base64Data,
              },
            });
            parts.push({
              text: `${EXTRACT_INSTRUCTION_PROMPT}\n\nRead all visible English text or vocabulary from the attached image carefully and extract the English vocabulary items with their Arabic meanings.`,
            });
          } else if (body.mode === 'file') {
            if (!body.base64Data) {
              res.status(400).json({ error: 'لم يتم استلام محتوى الملف.' });
              return;
            }
            const fileName = (body.fileName ?? '').toLowerCase();
            const mimeType = body.mimeType || 'application/octet-stream';
            const fileBuffer = Buffer.from(body.base64Data, 'base64');
            const keepLayout = body.preserveFormatting !== false;

            if (fileBuffer.length > 20 * 1024 * 1024) {
              res.status(400).json({ error: 'استخدم مسار الرفع المضغوط للملفات الكبيرة.' });
              return;
            }

            if (
              fileName.endsWith('.docx') ||
              mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ) {
              const result = await mammoth.extractRawText({ buffer: fileBuffer });
              const extractedDocText = (result.value ?? '').trim();
              if (!extractedDocText) {
                res.status(400).json({ error: 'لم يتم العثور على نص مقروء داخل ملف Word.' });
                return;
              }
              fallbackReadableText = extractedDocText;
              parts.push({
                text: `${EXTRACT_INSTRUCTION_PROMPT}\n\n${
                  keepLayout
                    ? 'Preserve the original section headings, lists, and reading order from the DOCX content, and also include a concise "preservedLayoutText" summary preserving headings/lists.'
                    : ''
                }\nExtracted text from DOCX file (${body.fileName}):\n"""\n${extractedDocText.slice(0, 25000)}\n"""`,
              });
            } else if (mimeType === 'application/pdf' || fileName.endsWith('.pdf')) {
              if (fileBuffer.subarray(0, 5).toString('ascii') !== '%PDF-') {
                res.status(400).json({ error: 'الملف المرفوع لا يبدو كملف PDF صالح.' });
                return;
              }
              fallbackReadableText = extractReadableTextFromPdfBuffer(fileBuffer);
              parts.push({
                inlineData: {
                  mimeType: 'application/pdf',
                  data: body.base64Data,
                },
              });
              parts.push({
                text: `${EXTRACT_INSTRUCTION_PROMPT}

PDF Formatting & Layout Rules:
- Read all pages of the attached PDF carefully${keepLayout ? ' while strictly preserving the document structure (headings, numbered lists, multi-column tables, word-to-meaning rows, and original sequence)' : ''}.
- If the PDF contains an English/Arabic vocabulary table or glossary, preserve each exact English term paired with its listed Arabic translation and context sentence in the original order.
- Also provide "preservedLayoutText": a clean, readable plain-text/Markdown reconstruction (up to 2000 characters) of the key PDF sections, headings, and lists preserving the original formatting.`,
              });
            } else if (mimeType.startsWith('image/')) {
              parts.push({
                inlineData: {
                  mimeType,
                  data: body.base64Data,
                },
              });
              parts.push({
                text: `${EXTRACT_INSTRUCTION_PROMPT}\n\nExtract the English vocabulary words from this image.`,
              });
            } else {
              const decodedText = fileBuffer.toString('utf-8').trim();
              if (!decodedText) {
                res.status(400).json({ error: 'الملف فارغ أو لا يحتوي على نص مقروء.' });
                return;
              }
              fallbackReadableText = decodedText;
              parts.push({
                text: `${EXTRACT_INSTRUCTION_PROMPT}\n\n${
                  keepLayout
                    ? 'Preserve headings, columns, and line structure from this file and include a concise formatted preview in "preservedLayoutText".'
                    : ''
                }\nContent from uploaded file (${body.fileName || 'document'}):\n"""\n${decodedText.slice(0, 25000)}\n"""`,
              });
            }
          } else {
            res.status(400).json({ error: 'وضع استخراج غير مدعوم.' });
            return;
          }

          if (useGemini) {
            try {
              const ai = getAiClient();
              const { parsed, cleanedWords } = await withTimeout(
                executeVocabExtraction(ai, parts),
                fallbackReadableText ? 8000 : 25000,
                'extract-vocab'
              );

              if (cleanedWords.length > 0) {
                const payload = {
                  sourceSummary:
                    parsed.sourceSummary || `تم استخراج ${cleanedWords.length} كلمة بنجاح.`,
                  preservedLayoutText:
                    typeof parsed.preservedLayoutText === 'string' &&
                    parsed.preservedLayoutText.trim()
                      ? parsed.preservedLayoutText.trim().slice(0, 3000)
                      : undefined,
                  words: cleanedWords,
                };

                if (textCacheKey) {
                  setBoundedCache(extractCache, textCacheKey, payload, MAX_EXTRACT_CACHE);
                }

                res.json(payload);
                return;
              }
            } catch (geminiErr) {
              if (!fallbackReadableText || fallbackReadableText.length < 2) {
                throw geminiErr;
              }
            }
          }

          if (fallbackReadableText && fallbackReadableText.length >= 2) {
            const specialty = normalizeSpecialty(body.specialty);
            const freeResult = await extractVocabularyFreeLocal(fallbackReadableText);
            if (freeResult.words.length === 0) {
              res.status(422).json({
                error: 'لم يتم العثور على كلمات إنجليزية واضحة في المحتوى المدخل. جرّب نصاً أوضح.',
              });
              return;
            }
            const adaptedWords =
              specialty !== 'general'
                ? freeResult.words.map((w) => {
                    const adapted = adaptCardContextForSpecialty(w.word, w.arabicMeaning, specialty);
                    return {
                      ...w,
                      exampleEn: adapted.exampleEn,
                      exampleAr: adapted.exampleAr,
                      usageContext: adapted.usageContext,
                    };
                  })
                : freeResult.words;
            const payload = {
              ...freeResult,
              words: adaptedWords,
              extractionMode: 'free-dictionary',
            };
            if (textCacheKey) {
              setBoundedCache(extractCache, textCacheKey, payload, MAX_EXTRACT_CACHE);
            }
            res.json(payload);
            return;
          }

          res.status(422).json({
            error:
              'لاستخراج الكلمات مجاناً بدون مفتاح، يرجى استخدام تبويب النص أو رفع ملف (TXT, DOCX, PDF نصي) يحتوي على كلمات إنجليزية.',
          });
        } catch (error: unknown) {
          handleExtractionError(res, error, '/api/extract-vocab');
        }
      });

      // Shared Community Vocabulary Library ("كلمات المستخدمين المحفوظة")
      // Aggregates shared words + words stored on every registered email (even after logout)
      app.get('/api/shared-words', (req, res) => {
        const query = typeof req.query.q === 'string' ? req.query.q : undefined;
        const emailWords = listAllRegisteredEmailWords();
        const words = listSharedWords(query, emailWords);
        res.json({ words, totalCount: words.length });
      });

      app.post('/api/shared-words', async (req, res) => {
        try {
          const auth = getAuthenticatedSession(req);
          if (auth?.viaCookie && !verifySameOriginOrHeader(req)) {
            res.status(403).json({ error: 'طلب غير مصرح به.' });
            return;
          }
          const authUser = auth ? findUserById(auth.session.uid) : null;
          const contributorName = authUser ? authUser.username : 'مستخدم';

          const body = (req.body || {}) as {
            words?: SharedWordCandidateInput[];
            validateWithAi?: boolean;
          };

          const rawList = Array.isArray(body.words) ? body.words.slice(0, 40) : [];
          if (rawList.length === 0) {
            res.status(400).json({ error: 'لا توجد كلمات لإضافتها.' });
            return;
          }

          const structurallyValid: Array<{
            word: string;
            wordLower: string;
            arabicMeaning: string;
            partOfSpeech: 'noun' | 'verb' | 'adjective' | 'adverb' | 'phrase' | 'other';
            phonetic: string;
            exampleEn: string;
            exampleAr: string;
            level: 'beginner' | 'intermediate' | 'advanced';
            deck: string;
          }> = [];

          for (const raw of rawList) {
            const check = validateSharedWordCandidate(raw);
            if (check.valid) {
              structurallyValid.push(check.cleaned);
            }
          }

          if (structurallyValid.length === 0) {
            res.json({ ok: true, addedCount: 0, updatedCount: 0, savedWords: [] });
            return;
          }

          let finalWordsToSave = structurallyValid;

          // For manually entered words, verify linguistic correctness & enrich IPA/examples via Gemini when available
          if (body.validateWithAi && hasConfiguredGeminiKey()) {
            try {
              const ai = getAiClient();
              const checkRes = await generateWithFreeFallback(ai, {
                contents: `Verify each English-Arabic vocabulary item submitted by a learner and classify its difficulty level strictly by word strength (CEFR).
For each item, check if "word" is a genuine English word or valid English phrase (not gibberish or typos) and if "arabicMeaning" is a valid Arabic translation for it.
Return isValid: true only for genuine valid English words/phrases with sensible Arabic meaning.
If valid, also provide accurate IPA "phonetic", a natural "exampleEn" (if missing), its Arabic translation "exampleAr" (if missing), and difficulty "level" strictly according to the word's strength:
- "beginner": Basic everyday A1-A2 vocabulary (e.g. house, eat, happy, fast, book).
- "intermediate": Workplace, study, and practical B1-B2 vocabulary (e.g. collaborate, deadline, efficient, strategy).
- "advanced": Strong academic, literary, technical, or C1-C2 vocabulary/idioms (e.g. resilient, paradigm, ubiquitous, mitigate, articulate).

Items:
${JSON.stringify(
  structurallyValid.map((w, idx) => ({
    index: idx,
    word: w.word,
    arabicMeaning: w.arabicMeaning,
    partOfSpeech: w.partOfSpeech,
    exampleEn: w.exampleEn,
  }))
)}`,
                config: {
                  thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
                  responseMimeType: 'application/json',
                  responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                      items: {
                        type: Type.ARRAY,
                        items: {
                          type: Type.OBJECT,
                          properties: {
                            index: { type: Type.INTEGER },
                            isValid: { type: Type.BOOLEAN },
                            correctedWord: { type: Type.STRING },
                            phonetic: { type: Type.STRING },
                            exampleEn: { type: Type.STRING },
                            exampleAr: { type: Type.STRING },
                            level: { type: Type.STRING },
                          },
                          required: ['index', 'isValid'],
                        },
                      },
                    },
                    required: ['items'],
                  },
                },
              });

              const parsed = JSON.parse((checkRes.text ?? '{}').trim()) as {
                items?: Array<{
                  index: number;
                  isValid: boolean;
                  correctedWord?: string;
                  phonetic?: string;
                  exampleEn?: string;
                  exampleAr?: string;
                  level?: string;
                }>;
              };

              if (Array.isArray(parsed.items)) {
                const verified: typeof structurallyValid = [];
                for (const ev of parsed.items) {
                  const orig = structurallyValid[ev.index];
                  if (!orig || !ev.isValid) continue;
                  const cleanWord =
                    ev.correctedWord && /^[A-Za-z][A-Za-z0-9\s'’.-]{0,79}$/.test(ev.correctedWord.trim())
                      ? ev.correctedWord.trim()
                      : orig.word;
                  verified.push({
                    ...orig,
                    word: cleanWord,
                    wordLower: cleanWord.toLowerCase(),
                    phonetic:
                      orig.phonetic && orig.phonetic !== `/${orig.wordLower}/`
                        ? orig.phonetic
                        : ev.phonetic?.trim() || orig.phonetic,
                    exampleEn: orig.exampleEn || ev.exampleEn?.trim() || '',
                    exampleAr: orig.exampleAr || ev.exampleAr?.trim() || '',
                    level: ['beginner', 'intermediate', 'advanced'].includes(
                      String(ev.level || '').toLowerCase()
                    )
                      ? (String(ev.level).toLowerCase() as 'beginner' | 'intermediate' | 'advanced')
                      : orig.level,
                  });
                }
                finalWordsToSave = verified;
              }
            } catch {
              // Fallback to structural validation if AI verification is transiently unavailable
              finalWordsToSave = structurallyValid;
            }
          }

          if (finalWordsToSave.length === 0) {
            res.json({ ok: true, addedCount: 0, updatedCount: 0, savedWords: [] });
            return;
          }

          const result = upsertSharedWords(finalWordsToSave, contributorName);
          res.status(201).json({
            ok: true,
            addedCount: result.addedCount,
            updatedCount: result.updatedCount,
            savedWords: result.savedWords,
          });
        } catch (error) {
          console.error('Error in POST /api/shared-words:', error);
          res.status(500).json({ error: 'تعذر حفظ الكلمات في المكتبة المشتركة حالياً.' });
        }
      });

      // Classify shared words strictly by lexical strength (Advanced -> Intermediate -> Beginner)
      app.post('/api/shared-words/reclassify', async (_req, res) => {
        try {
          const emailWords = listAllRegisteredEmailWords();
          // Ensure any words currently living only inside registered email files are materialized
          // in the shared index so updated levels persist cleanly
          for (const entry of emailWords) {
            const check = validateSharedWordCandidate(entry.candidate);
            if (check.valid) {
              syncMissingSharedWords([check.cleaned], entry.addedBy);
            }
          }

          const currentWords = listSharedWords(undefined, emailWords);
          if (currentWords.length === 0) {
            res.json({ ok: true, updatedCount: 0, words: [] });
            return;
          }

          let totalUpdated = 0;

          if (hasConfiguredGeminiKey()) {
            try {
              const ai = getAiClient();
              // Process in batches of up to 40 words
              for (let offset = 0; offset < currentWords.length; offset += 40) {
                const batch = currentWords.slice(offset, offset + 40);
                const classifyRes = await withTimeout(
                  generateWithFreeFallback(ai, {
                    contents: `Classify each English vocabulary word strictly according to its lexical strength and CEFR difficulty:
- "beginner": Basic everyday A1-A2 words (e.g., book, happy, go, water, house, friend).
- "intermediate": Practical B1-B2 words used in work, study, and news (e.g., collaborate, efficient, deadline, strategy, opportunity).
- "advanced": Strong C1-C2 academic, formal, literary, technical, or rare expressive words (e.g., resilient, paradigm, ubiquitous, articulate, mitigate, unprecedented).

Words to classify:
${JSON.stringify(batch.map((w) => ({ id: w.id, word: w.word, arabicMeaning: w.arabicMeaning })))}`,
                    config: {
                      thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
                      responseMimeType: 'application/json',
                      responseSchema: {
                        type: Type.OBJECT,
                        properties: {
                          classifications: {
                            type: Type.ARRAY,
                            items: {
                              type: Type.OBJECT,
                              properties: {
                                id: { type: Type.STRING },
                                level: { type: Type.STRING },
                              },
                              required: ['id', 'level'],
                            },
                          },
                        },
                        required: ['classifications'],
                      },
                    },
                  }),
                  5000,
                  'reclassify'
                );

                const parsed = JSON.parse((classifyRes.text ?? '{}').trim()) as {
                  classifications?: Array<{ id: string; level: string }>;
                };

                const validUpdates = (parsed.classifications ?? [])
                  .filter((c) => ['beginner', 'intermediate', 'advanced'].includes(String(c.level).toLowerCase()))
                  .map((c) => ({
                    id: c.id,
                    level: String(c.level).toLowerCase() as 'beginner' | 'intermediate' | 'advanced',
                  }));

                totalUpdated += updateSharedWordLevels(validUpdates);
              }
            } catch {
              // Fall back seamlessly to built-in free lexical strength classifier
              const localUpdates = currentWords.map((w) => ({
                id: w.id,
                level: estimateWordStrengthFallback(w.word),
              }));
              totalUpdated = updateSharedWordLevels(localUpdates);
            }
          } else {
            const localUpdates = currentWords.map((w) => ({
              id: w.id,
              level: estimateWordStrengthFallback(w.word),
            }));
            totalUpdated = updateSharedWordLevels(localUpdates);
          }

          res.json({
            ok: true,
            updatedCount: totalUpdated,
            words: listSharedWords(undefined, listAllRegisteredEmailWords()),
          });
        } catch (error) {
          console.error('Error in POST /api/shared-words/reclassify:', error);
          res.status(500).json({ error: 'تعذر تحديث تصنيف الكلمات حالياً.' });
        }
      });

      // Classify any list of personal words by lexical strength
      app.post('/api/words/classify-strength', async (req, res) => {
        try {
          const body = (req.body || {}) as {
            words?: Array<{ id: string; word: string; arabicMeaning?: string }>;
          };
          const list = Array.isArray(body.words) ? body.words.slice(0, 60) : [];
          if (list.length === 0) {
            res.json({ classifications: [] });
            return;
          }

          if (hasConfiguredGeminiKey()) {
            try {
              const ai = getAiClient();
              const classifyRes = await withTimeout(
                generateWithFreeFallback(ai, {
                  contents: `Classify each English word strictly by lexical strength and CEFR level:
- "beginner": Basic everyday A1-A2 words (e.g., book, water, happy, walk, friend).
- "intermediate": Practical B1-B2 words used in work, study, and daily discussion (e.g., collaborate, efficient, deadline, strategy).
- "advanced": Strong C1-C2 academic, formal, literary, technical, or rare expressive words (e.g., resilient, paradigm, ubiquitous, articulate, mitigate).

Words:
${JSON.stringify(list.map((w) => ({ id: w.id, word: w.word, arabicMeaning: w.arabicMeaning || '' })))}`,
                  config: {
                    thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
                    responseMimeType: 'application/json',
                    responseSchema: {
                      type: Type.OBJECT,
                      properties: {
                        classifications: {
                          type: Type.ARRAY,
                          items: {
                            type: Type.OBJECT,
                            properties: {
                              id: { type: Type.STRING },
                              level: { type: Type.STRING },
                            },
                            required: ['id', 'level'],
                          },
                        },
                      },
                      required: ['classifications'],
                    },
                  },
                }),
                5000,
                'classify-strength'
              );

              const parsed = JSON.parse((classifyRes.text ?? '{}').trim()) as {
                classifications?: Array<{ id: string; level: string }>;
              };

              const classifications = (parsed.classifications ?? [])
                .filter((c) => ['beginner', 'intermediate', 'advanced'].includes(String(c.level).toLowerCase()))
                .map((c) => ({
                  id: c.id,
                  level: String(c.level).toLowerCase() as 'beginner' | 'intermediate' | 'advanced',
                }));

              if (classifications.length > 0) {
                res.json({ classifications });
                return;
              }
            } catch {
              // Fall back cleanly to built-in free lexical strength classifier
            }
          }

          res.json({
            classifications: list.map((w) => ({
              id: w.id,
              level: estimateWordStrengthFallback(w.word),
            })),
          });
        } catch (error) {
          console.error('Error in POST /api/words/classify-strength:', error);
          res.status(500).json({ error: 'تعذر تصنيف قوة الكلمات حالياً.' });
        }
      });

  app.post('/api/tts', async (req, res) => {
    try {
      const {
        text,
        voice = 'Kore',
        slow = false,
        dialect = 'us',
        language = 'en',
      } = req.body as {
        text?: string;
        voice?: string;
        slow?: boolean;
        dialect?: 'us' | 'uk';
        language?: 'en' | 'ar' | 'bilingual';
      };

      const cleanText = (text ?? '').trim();
      if (!cleanText) {
        res.status(400).json({ error: 'النص المطلوب نطقه فارغ.' });
        return;
      }
      if (cleanText.length > 950) {
        res.status(400).json({ error: 'النص طويل جداً للنطق الفوري (الحد الأقصى 950 حرف).' });
        return;
      }

      const selectedVoice = VALID_VOICES.has(voice) ? voice : 'Kore';
      const resolvedDialect: 'us' | 'uk' = dialect === 'uk' ? 'uk' : 'us';
      const resolvedLanguage: 'en' | 'ar' | 'bilingual' =
        language === 'ar' || language === 'bilingual' ? language : 'en';
      const cacheKey = `${selectedVoice}::${resolvedDialect}::${resolvedLanguage}::${slow ? 'slow' : 'normal'}::${cleanText.toLowerCase()}`;
      const cachedAudio = ttsCache.get(cacheKey);
      if (cachedAudio) {
        res.json({ audioBase64: cachedAudio, mimeType: 'audio/wav', cached: true });
        return;
      }

      const ai = getAiClient();
      const accentDesc =
        resolvedDialect === 'uk'
          ? 'Standard British Received Pronunciation (UK) English'
          : 'General American (US) English';
      const stylePrompt =
        resolvedLanguage === 'bilingual'
          ? slow
            ? `Speak warmly and slowly as a friendly bilingual English-Arabic language tutor, pronouncing ${accentDesc} sentences clearly and Arabic explanations naturally.`
            : `Speak warmly and naturally as a friendly bilingual English-Arabic language tutor, pronouncing ${accentDesc} sentences cleanly and Arabic explanations smoothly.`
          : resolvedLanguage === 'ar'
            ? 'Speak warmly and clearly in modern standard Arabic suitable for a language learner.'
            : slow
              ? `Speak in clear, slow, carefully articulated ${accentDesc} pronunciation suitable for a language learner.`
              : `Speak in clear, natural, well-paced ${accentDesc} pronunciation for a language learner.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash-lite-tts',
        contents: [
          {
            role: 'user',
            parts: [
              {
                text: cleanText,
                speechMetadata: {
                  style: stylePrompt,
                },
              },
            ],
          },
        ],
        config: {
          responseModalities: ['AUDIO'],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: selectedVoice },
            },
          },
        },
      });

      const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
      if (!base64Audio) {
        res.status(502).json({ error: 'لم يتم استلام ملف صوتي من خدمة النطق.' });
        return;
      }

      setTtsCache(cacheKey, base64Audio);
      res.json({
        audioBase64: base64Audio,
        mimeType: 'audio/wav',
        cached: false,
      });
    } catch (error) {
      console.error('Error in /api/tts:', error);
      const classified = classifyGeminiHttpError(error);
      if (classified.isRateLimit) {
        res.status(429).json({
          error: 'تم تجاوز حد الطلبات المؤقت المجاني. سيتم استخدام نطق المتصفح تلقائياً.',
        });
        return;
      }
      res.status(500).json({
        error: 'تعذر توليد النطق الصوتي عبر الخادم حالياً.',
      });
    }
  });

  // 3. Pronunciation Evaluation using Gemini Audio Transcription + Coaching Feedback
  app.post('/api/pronunciation-check', async (req, res) => {
    try {
      const { targetText, audioBase64, mimeType } = req.body as {
        targetText?: string;
        audioBase64?: string;
        mimeType?: string;
      };

      const cleanTarget = (targetText ?? '').trim();
      if (!cleanTarget || !audioBase64) {
        res.status(400).json({ error: 'يرجى تسجيل صوتك أولاً للمقارنة.' });
        return;
      }

      const cleanMime = (mimeType || 'audio/webm').split(';')[0].trim();
      const ai = getAiClient();

      // Transcribe user speech using gemini-3.5-transcribe
      const transcribeRes = await ai.models.generateContent({
        model: 'gemini-3.5-transcribe',
        contents: [
          {
            role: 'user',
            parts: [
              {
                inlineData: {
                  mimeType: cleanMime,
                  data: audioBase64,
                },
              },
              {
                text: 'Transcribe the spoken English audio accurately. Output only the English words heard, or "(silence)" if no speech was heard.',
              },
            ],
          },
        ],
      });

      const transcript = (transcribeRes.text ?? '').trim();

      if (!transcript || transcript.toLowerCase().includes('silence')) {
        res.json({
          transcript: '',
          score: 0,
          matched: false,
          feedbackAr: 'لم نتمكن من سماع صوت واضح في التسجيل. قرّب الميكروفون وحاول النطق بصوت واضح.',
        });
        return;
      }

      // Compare transcript with targetText and give helpful Arabic pronunciation coaching
      const evalRes = await generateWithFreeFallback(ai, {
        contents: `Compare a language learner's spoken English transcription against the target phrase and return an accurate score (0-100) and concise Arabic coaching feedback.
Target English text: "${cleanTarget}"
Transcribed user speech: "${transcript}"

Rules:
- Ignore minor capitalization/punctuation differences.
- If the words match closely or phonetically identically, give 90-100 and matched=true.
- If some words differ or are missing, give proportional score (e.g. 50-80) and explain gently in Arabic which word needs clearer articulation.
- Keep feedbackAr to 1-2 encouraging Arabic sentences.`,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              score: { type: Type.INTEGER },
              matched: { type: Type.BOOLEAN },
              feedbackAr: { type: Type.STRING },
            },
            required: ['score', 'matched', 'feedbackAr'],
          },
        },
      });

      const evalParsed = JSON.parse((evalRes.text ?? '{}').trim()) as {
        score?: number;
        matched?: boolean;
        feedbackAr?: string;
      };

      res.json({
        transcript,
        score: Math.max(0, Math.min(100, Number(evalParsed.score ?? 0))),
        matched: Boolean(evalParsed.matched),
        feedbackAr: evalParsed.feedbackAr || 'استمع للنطق الصحيح وقارنه بتسجيلك ثم أعد المحاولة.',
      });
    } catch (error) {
      console.error('Error in /api/pronunciation-check:', error);
      const classified = classifyGeminiHttpError(error);
      if (classified.isRateLimit) {
        res.status(429).json({
          error: 'تم تجاوز حد الطلبات المؤقت المجاني. يرجى الانتظار بضع لحظات ثم المحاولة مجدداً.',
        });
        return;
      }
      res.status(500).json({
        error: 'تعذر تحليل التسجيل الصوتي حالياً. تأكد من صلاحية الميكروفون وحاول مجدداً.',
      });
    }
  });

  // 3b. Reading Practice Passage Generator (Articles & Sentences with curated fallback)
  const FALLBACK_READING_PASSAGES: Array<{
    mode: 'article' | 'sentences';
    level: 'beginner' | 'intermediate' | 'advanced';
    titleEn: string;
    titleAr: string;
    topic: string;
    sentences: Array<{ en: string; ar: string }>;
  }> = [
    {
      mode: 'article',
      level: 'beginner',
      titleEn: 'A Calm Morning Routine',
      titleAr: 'روتين صباحي هادئ',
      topic: 'الحياة اليومية',
      sentences: [
        { en: 'Every morning, I wake up early and open the window for fresh air.', ar: 'كل صباح أستيقظ مبكراً وأفتح النافذة لدخول الهواء النقي.' },
        { en: 'I drink a glass of warm water before eating a healthy breakfast.', ar: 'أشرب كوباً من الماء الدافئ قبل تناول فطور صحي.' },
        { en: 'Then I read ten pages from an English book to improve my vocabulary.', ar: 'بعد ذلك أقرأ عشر صفحات من كتاب إنجليزي لتطوير مفرداتي.' },
        { en: 'This simple habit gives me energy and focus for the whole day.', ar: 'هذه العادة البسيطة تمنحني الطاقة والتركيز طوال اليوم.' },
      ],
    },
    {
      mode: 'article',
      level: 'intermediate',
      titleEn: 'Learning a Language Through Stories',
      titleAr: 'تعلم اللغة من خلال القصص',
      topic: 'التعليم والتطوير',
      sentences: [
        { en: 'Reading short articles aloud is one of the fastest ways to build speaking confidence.', ar: 'قراءة المقالات القصيرة بصوت مسموع من أسرع الطرق لبناء الثقة في التحدث.' },
        { en: 'When you hear your own voice, your brain notices how words connect naturally.', ar: 'عندما تسمع صوتك، يلاحظ عقلك كيف ترتبط الكلمات ببعضها بشكل طبيعي.' },
        { en: 'Do not worry about making small pronunciation mistakes at the beginning.', ar: 'لا تقلق بشأن ارتكاب أخطاء نطق بسيطة في البداية.' },
        { en: 'Each correction helps you speak more clearly in real conversations.', ar: 'كل تصحيح يساعدك على التحدث بوضوح أكبر في المحادثات الحقيقية.' },
      ],
    },
    {
      mode: 'article',
      level: 'advanced',
      titleEn: 'Resilience in a Fast-Changing World',
      titleAr: 'المرونة في عالم سريع التغير',
      topic: 'التقنية والعمل',
      sentences: [
        { en: 'Modern technology continuously reshapes how professionals collaborate across continents.', ar: 'تعيد التقنية الحديثة تشكيل كيفية تعاون المهنيين عبر القارات باستمرار.' },
        { en: 'Adaptability and clear communication have become essential skills for long-term success.', ar: 'أصبحت القدرة على التكيف والتواصل الواضح مهارتين أساسيتين للنجاح طويل الأمد.' },
        { en: 'Those who embrace lifelong learning turn unfamiliar challenges into valuable opportunities.', ar: 'من يتبنون التعلم مدى الحياة يحولون التحديات غير المألوفة إلى فرص قيمة.' },
        { en: 'Consistent daily practice transforms complex ideas into effortless fluency.', ar: 'الممارسة اليومية المستمرة تحول الأفكار المعقدة إلى طلاقة سلسة.' },
      ],
    },
    {
      mode: 'sentences',
      level: 'beginner',
      titleEn: 'Everyday Reading Sentences',
      titleAr: 'جمل يومية للتدرب على القراءة',
      topic: 'الحياة اليومية',
      sentences: [
        { en: 'Could you please speak a little slower?', ar: 'هل يمكنك التحدث ببطء أكثر من فضلك؟' },
        { en: 'I enjoy practicing my English pronunciation every evening.', ar: 'أستمتع بالتدرب على نطقي للإنجليزية كل مساء.' },
        { en: 'The train arrives at the central station at eight o’clock.', ar: 'يصل القطار إلى المحطة المركزية في الساعة الثامنة.' },
        { en: 'Thank you very much for your kind help today.', ar: 'شكراً جزيلاً لك على مساعدتك اللطيفة اليوم.' },
        { en: 'Let us meet tomorrow morning at the coffee shop.', ar: 'دعنا نلتقي صباح الغد في المقهى.' },
      ],
    },
    {
      mode: 'sentences',
      level: 'intermediate',
      titleEn: 'Practical Communication Sentences',
      titleAr: 'جمل عملية للتواصل والطلاقة',
      topic: 'العمل والسفر',
      sentences: [
        { en: 'Consistent practice makes difficult words feel natural over time.', ar: 'الممارسة المستمرة تجعل الكلمات الصعبة تبدو طبيعية مع الوقت.' },
        { en: 'We should review our schedule before the afternoon meeting starts.', ar: 'يجب أن نراجع جدولنا قبل بدء اجتماع بعد الظهر.' },
        { en: 'I noticed a slight improvement in my reading speed this week.', ar: 'لاحظت تحسناً طفيفاً في سرعة قراءتي هذا الأسبوع.' },
        { en: 'Please let me know if you need any further explanation.', ar: 'يرجى إعلامي إذا كنت بحاجة إلى أي توضيح إضافي.' },
        { en: 'Traveling abroad encourages people to use foreign languages bravely.', ar: 'السفر للخارج يشجع الناس على استخدام اللغات الأجنبية بشجاعة.' },
      ],
    },
    {
      mode: 'sentences',
      level: 'advanced',
      titleEn: 'Articulate Expression Sentences',
      titleAr: 'جمل متقدمة لإتقان المخارج والطلاقة',
      topic: 'التعبير المتقدم',
      sentences: [
        { en: 'Clear articulation requires careful attention to vowel length and word stress.', ar: 'يتطلب النطق الواضح انتباهاً دقيقاً لطول حروف العلة ونبر الكلمات.' },
        { en: 'Effective leaders articulate their vision with simplicity and precision.', ar: 'يعبّر القادة الفعّالون عن رؤيتهم ببساطة ودقة.' },
        { en: 'Throughout the conference, speakers emphasized sustainable innovation.', ar: 'خلال المؤتمر، شدد المتحدثون على الابتكار المستدام.' },
        { en: 'Thoughtful feedback accelerates mastery far better than silent repetition.', ar: 'التغذية الراجعة المدروسة تسرّع الإتقان أفضل بكثير من التكرار الصامت.' },
        { en: 'Mastering subtle rhythm patterns makes spoken English sound effortless.', ar: 'إتقان أنماط الإيقاع الدقيقة يجعل الإنجليزية المنطوقة تبدو سلسة.' },
      ],
    },
  ];

  app.post('/api/reading/generate', async (req, res) => {
    try {
      const {
        mode = 'article',
        level = 'intermediate',
        topic = 'الحياة اليومية',
        words = [],
      } = (req.body || {}) as {
        mode?: 'article' | 'sentences';
        level?: 'beginner' | 'intermediate' | 'advanced';
        topic?: string;
        words?: Array<{ word: string; arabicMeaning: string }>;
      };

      const cleanMode = mode === 'sentences' ? 'sentences' : 'article';
      const cleanLevel =
        level === 'beginner' || level === 'advanced' ? level : 'intermediate';
      const cleanTopic = String(topic || 'الحياة اليومية').trim().slice(0, 60);
      const userWordsSample = Array.isArray(words)
        ? words
            .slice(0, 6)
            .map((w) => String(w.word || '').trim())
            .filter(Boolean)
        : [];

      if (hasConfiguredGeminiKey()) {
        try {
          const ai = getAiClient();
          const styleHint =
            cleanMode === 'article'
              ? 'Write a cohesive, engaging short English article of 4 to 5 well-connected sentences (10-16 words per sentence) suitable for reading aloud.'
              : 'Write 5 distinct, natural standalone English sentences (8-14 words each) suitable for pronunciation and reading-aloud practice.';
          const vocabHint =
            userWordsSample.length > 0
              ? `Naturally weave in 2 to 4 of these learner words if appropriate: ${userWordsSample.join(', ')}.`
              : '';

          const prompt = `You are an expert English reading and pronunciation coach for Arabic speakers.
${styleHint}
Difficulty level: ${cleanLevel}
Topic preference: ${cleanTopic}
${vocabHint}
Avoid tricky proper nouns, rare abbreviations, or symbols that are ambiguous when read aloud.
Return JSON with:
- titleEn: concise English title
- titleAr: natural Arabic title
- sentences: array of objects with { en: English sentence, ar: accurate Arabic translation }`;

          const aiRes = await generateWithFreeFallback(ai, {
            contents: prompt,
            config: {
              thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  titleEn: { type: Type.STRING },
                  titleAr: { type: Type.STRING },
                  sentences: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        en: { type: Type.STRING },
                        ar: { type: Type.STRING },
                      },
                      required: ['en', 'ar'],
                    },
                  },
                },
                required: ['titleEn', 'titleAr', 'sentences'],
              },
            },
          });

          const parsed = JSON.parse((aiRes.text ?? '{}').trim()) as {
            titleEn?: string;
            titleAr?: string;
            sentences?: Array<{ en?: string; ar?: string }>;
          };

          const cleanSentences = (parsed.sentences ?? [])
            .map((s) => ({
              en: String(s.en ?? '').trim(),
              ar: String(s.ar ?? '').trim(),
            }))
            .filter((s) => s.en.length > 0 && s.ar.length > 0)
            .slice(0, 6);

          if (cleanSentences.length >= 2) {
            res.json({
              passage: {
                id: `read-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
                titleEn: (parsed.titleEn || 'Reading Practice').trim(),
                titleAr: (parsed.titleAr || 'نص تدريبي للقراءة').trim(),
                mode: cleanMode,
                level: cleanLevel,
                topic: cleanTopic,
                sentences: cleanSentences,
              },
            });
            return;
          }
        } catch (aiErr) {
          console.warn('Gemini reading generation failed, using curated fallback:', aiErr);
        }
      }

      const candidates = FALLBACK_READING_PASSAGES.filter(
        (p) => p.mode === cleanMode && p.level === cleanLevel
      );
      const chosen =
        candidates[Math.floor(Math.random() * candidates.length)] ||
        FALLBACK_READING_PASSAGES.find((p) => p.mode === cleanMode) ||
        FALLBACK_READING_PASSAGES[0];

      res.json({
        passage: {
          id: `read-fallback-${Date.now()}`,
          titleEn: chosen.titleEn,
          titleAr: chosen.titleAr,
          mode: chosen.mode,
          level: chosen.level,
          topic: chosen.topic,
          sentences: chosen.sentences,
        },
      });
    } catch (error) {
      console.error('Error in POST /api/reading/generate:', error);
      res.status(500).json({ error: 'تعذر تجهيز نص القراءة حالياً.' });
    }
  });

  // 3c. Spoken Reading Evaluation & Red-Highlight Pronunciation Error Correction
  app.post('/api/reading/evaluate', async (req, res) => {
    try {
      const { targetText, audioBase64, mimeType } = (req.body || {}) as {
        targetText?: string;
        audioBase64?: string;
        mimeType?: string;
      };

      const cleanTarget = String(targetText ?? '').trim().slice(0, 2500);
      if (!cleanTarget || !audioBase64) {
        res.status(400).json({ error: 'يرجى تسجيل قراءتك بصوتك أولاً للتصحيح.' });
        return;
      }

      const cleanMime = String(mimeType || 'audio/webm').split(';')[0].trim();
      const ai = getAiClient();

      // Transcribe spoken English verbatim without silently fixing mistakes
      const transcribeRes = await ai.models.generateContent({
        model: 'gemini-3.5-transcribe',
        contents: [
          {
            role: 'user',
            parts: [
              {
                inlineData: {
                  mimeType: cleanMime,
                  data: audioBase64,
                },
              },
              {
                text: 'Transcribe the spoken English audio verbatim exactly as heard. Do not auto-correct mispronounced or skipped words; transcribe the closest English words actually spoken. Output only the spoken English words, or "(silence)" if no speech was heard.',
              },
            ],
          },
        ],
      });

      const transcript = (transcribeRes.text ?? '').trim();
      if (!transcript || transcript.toLowerCase().includes('silence')) {
        const emptyAlign = alignSpokenReading(cleanTarget, '');
        res.json({
          transcript: '',
          score: 0,
          matched: false,
          summaryAr:
            'لم نتمكن من سماع صوت واضح في التسجيل. قرّب الميكروفون واقرأ النص بصوت مسموع وواضح.',
          alignedTokens: emptyAlign.alignedTokens,
          corrections: emptyAlign.corrections,
          extraSpokenWords: [],
        });
        return;
      }

      const aligned = alignSpokenReading(cleanTarget, transcript);
      let summaryAr = aligned.matched
        ? 'ممتاز! قرأت النص بنطق سليم وواضح دون أخطاء لفظية.'
        : `أحسنت المحاولة! نطقت معظم الكلمات بوضوح، وهناك ${aligned.corrections.length} ${
            aligned.corrections.length === 1 ? 'كلمة تحتاج' : 'كلمات تحتاج'
          } تصحيحاً باللون الأحمر أدناه.`;

      const tipsMap = new Map<string, string>();

      // Enrich word-level pronunciation tips with Gemini when mistakes exist
      if (aligned.corrections.length > 0 || aligned.score < 100) {
        try {
          const errorSummary = aligned.corrections
            .slice(0, 10)
            .map((c) =>
              c.kind === 'missing'
                ? `Missing/skipped: "${c.expectedWord}"`
                : `Expected "${c.expectedWord}", heard "${c.spokenAs}"`
            )
            .join('; ');

          const coachRes = await generateWithFreeFallback(ai, {
            contents: `You are an encouraging English pronunciation coach for Arabic-speaking learners.
Target text: "${cleanTarget}"
Verbatim spoken transcript: "${transcript}"
Detected differences: ${errorSummary || 'Minor pronunciation hesitation'}

Provide:
1. summaryAr: 1-2 encouraging Arabic sentences evaluating the reading clarity.
2. wordTips: for up to 8 affected target words, give a short, practical Arabic pronunciation tip explaining how to pronounce the word correctly (e.g., silent letters, vowel sound, or articulation).`,
            config: {
              thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
              responseMimeType: 'application/json',
              responseSchema: {
                type: Type.OBJECT,
                properties: {
                  summaryAr: { type: Type.STRING },
                  wordTips: {
                    type: Type.ARRAY,
                    items: {
                      type: Type.OBJECT,
                      properties: {
                        expectedWord: { type: Type.STRING },
                        tipAr: { type: Type.STRING },
                      },
                      required: ['expectedWord', 'tipAr'],
                    },
                  },
                },
                required: ['summaryAr', 'wordTips'],
              },
            },
          });

          const coachParsed = JSON.parse((coachRes.text ?? '{}').trim()) as {
            summaryAr?: string;
            wordTips?: Array<{ expectedWord?: string; tipAr?: string }>;
          };

          if (coachParsed.summaryAr?.trim()) {
            summaryAr = coachParsed.summaryAr.trim();
          }
          for (const item of coachParsed.wordTips ?? []) {
            const key = String(item.expectedWord ?? '')
              .toLowerCase()
              .replace(/[^a-z0-9']/g, '');
            const tip = String(item.tipAr ?? '').trim();
            if (key && tip) {
              tipsMap.set(key, tip);
            }
          }
        } catch (coachErr) {
          console.warn('Gemini reading coaching skipped, returning aligned word errors:', coachErr);
        }
      }

      const enrichedTokens = aligned.alignedTokens.map((tok) => {
        if (tok.kind === 'correct') return tok;
        const key = tok.expectedWord.toLowerCase().replace(/[^a-z0-9']/g, '');
        const tipAr = tipsMap.get(key);
        return tipAr ? { ...tok, tipAr } : tok;
      });

      const enrichedCorrections = aligned.corrections.map((c) => {
        const key = c.expectedWord.toLowerCase().replace(/[^a-z0-9']/g, '');
        const tipAr = tipsMap.get(key);
        return tipAr ? { ...c, correctionTipAr: tipAr } : c;
      });

      res.json({
        transcript,
        score: aligned.score,
        matched: aligned.matched,
        summaryAr,
        alignedTokens: enrichedTokens,
        corrections: enrichedCorrections,
        extraSpokenWords: aligned.extraSpokenWords,
      });
    } catch (error) {
      console.error('Error in POST /api/reading/evaluate:', error);
      const classified = classifyGeminiHttpError(error);
      if (classified.isRateLimit) {
        res.status(429).json({
          error: 'تم تجاوز حد الطلبات المؤقت المجاني. يرجى الانتظار بضع لحظات ثم المحاولة مجدداً.',
        });
        return;
      }
      res.status(500).json({
        error: 'تعذر تحليل قراءتك الصوتية حالياً. تأكد من صلاحية الميكروفون وحاول مجدداً.',
      });
    }
  });

  // 4. Sentence Ordering Exercise Generator
  app.post('/api/exercises/sentences', async (req, res) => {
    try {
      const { words = [] } = (req.body || {}) as {
        words?: Array<{ word: string; arabicMeaning: string }>;
      };
      const ai = getAiClient();

      const focusList = words
        .slice(0, 8)
        .map((w) => `${w.word} (${w.arabicMeaning})`)
        .join(', ');

      const prompt = `Create 6 practical, natural English sentences (5 to 9 words each) for a sentence-reordering exercise for Arabic learners.
${focusList ? `Incorporate these learner words where natural: ${focusList}` : 'Use everyday useful English vocabulary.'}

For each sentence provide:
- id: unique string
- sentenceEn: clean natural English sentence without extra quotes (standard punctuation at end).
- sentenceAr: accurate Arabic translation.
- focusWord: the key vocabulary word highlighted in the sentence.`;

      const response = await generateWithFreeFallback(ai, {
        contents: prompt,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              sentences: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    id: { type: Type.STRING },
                    sentenceEn: { type: Type.STRING },
                    sentenceAr: { type: Type.STRING },
                    focusWord: { type: Type.STRING },
                  },
                  required: ['id', 'sentenceEn', 'sentenceAr', 'focusWord'],
                },
              },
            },
            required: ['sentences'],
          },
        },
      });

      const parsed = JSON.parse((response.text ?? '{}').trim()) as {
        sentences?: Array<{
          id: string;
          sentenceEn: string;
          sentenceAr: string;
          focusWord: string;
        }>;
      };

      const exercises = (parsed.sentences ?? []).map((s, idx) => {
        const tokens = s.sentenceEn
          .trim()
          .split(/\s+/)
          .filter(Boolean);
        return {
          id: s.id || `sent-${idx + 1}`,
          sentenceEn: s.sentenceEn.trim(),
          sentenceAr: s.sentenceAr.trim(),
          focusWord: s.focusWord.trim(),
          tokens,
        };
      });

      res.json({ exercises });
    } catch (error) {
      console.error('Error in /api/exercises/sentences:', error);
      const classified = classifyGeminiHttpError(error);
      if (classified.isRateLimit) {
        res.status(429).json({
          error: 'تم تجاوز حد الطلبات المؤقت المجاني. يرجى الانتظار بضع لحظات ثم المحاولة مجدداً.',
        });
        return;
      }
      res.status(500).json({
        error: 'تعذر تجهيز تمارين ترتيب الجمل حالياً.',
      });
    }
  });

  // 5. Daily Challenge Generator
  app.post('/api/daily-challenge/generate', async (req, res) => {
    try {
      const {
        source = 'topic',
        topic = 'travel',
        userWords = [],
        level = 'intermediate',
        dateKey,
      } = (req.body || {}) as {
        source?: 'user_words' | 'topic';
        topic?: 'travel' | 'work' | 'food' | 'daily';
        userWords?: Array<{ word: string; arabicMeaning: string }>;
        level?: string;
        dateKey?: string;
      };

      const todayStr = dateKey || new Date().toISOString().slice(0, 10);
      const dailyCacheKey =
        source === 'topic' ? `${todayStr}::${topic}::${level}` : null;
      if (dailyCacheKey) {
        const cachedDaily = dailyChallengeCache.get(dailyCacheKey);
        if (cachedDaily) {
          res.json(cachedDaily);
          return;
        }
      }

      const ai = getAiClient();
      const topicNames: Record<string, string> = {
        travel: 'Travel & Airports (السفر والمطارات)',
        work: 'Workplace & Meetings (العمل والاجتماعات)',
        food: 'Food & Dining (الطعام والمطاعم)',
        daily: 'Everyday Conversations (المحادثة اليومية)',
      };

      const userWordsText = userWords
        .slice(0, 12)
        .map((w) => `${w.word} = ${w.arabicMeaning}`)
        .join('; ');

      const contextInstruction =
        source === 'user_words' && userWords.length >= 3
          ? `Focus the challenge primarily on reviewing the learner's own saved words: ${userWordsText}. Also introduce 3 related bonus vocabulary words.`
          : `Focus the challenge on the theme: ${topicNames[topic] || topicNames.daily} tailored around ${level} level. Introduce 3 useful themed vocabulary words in "newWords" and build the tasks around this theme.`;

      const prompt = `Create a short, fun, engaging Daily English Challenge for an Arabic-speaking learner.
${contextInstruction}

Generate:
1. titleAr: Engaging short Arabic title for today's challenge.
2. themeLabelAr: Short Arabic theme label.
3. tasks: EXACTLY 5 quick tasks:
   - 2 "choice" tasks (pick correct word/meaning, 4 options)
   - 2 "fill_blank" tasks (sentence with _____, 4 options for fast fun tapping OR clear single-word answer; provide 4 options so mobile/desktop taps feel snappy!)
   - 1 "speak" task (a practical short English phrase to pronounce aloud; promptEn is the phrase to say, correctAnswer is the same phrase).
4. newWords: 3 fresh, useful English vocabulary words (with arabicMeaning, partOfSpeech, phonetic IPA, exampleEn, exampleAr, level) that the learner can save to their deck.`;

      const response = await generateWithFreeFallback(ai, {
        contents: prompt,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              titleAr: { type: Type.STRING },
              themeLabelAr: { type: Type.STRING },
              tasks: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    id: { type: Type.STRING },
                    type: { type: Type.STRING },
                    promptAr: { type: Type.STRING },
                    promptEn: { type: Type.STRING },
                    options: {
                      type: Type.ARRAY,
                      items: { type: Type.STRING },
                    },
                    correctAnswer: { type: Type.STRING },
                    tipAr: { type: Type.STRING },
                  },
                  required: [
                    'id',
                    'type',
                    'promptAr',
                    'promptEn',
                    'correctAnswer',
                    'tipAr',
                  ],
                },
              },
              newWords: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    word: { type: Type.STRING },
                    arabicMeaning: { type: Type.STRING },
                    partOfSpeech: { type: Type.STRING },
                    phonetic: { type: Type.STRING },
                    exampleEn: { type: Type.STRING },
                    exampleAr: { type: Type.STRING },
                    level: { type: Type.STRING },
                  },
                  required: [
                    'word',
                    'arabicMeaning',
                    'partOfSpeech',
                    'phonetic',
                    'exampleEn',
                    'exampleAr',
                    'level',
                  ],
                },
              },
            },
            required: ['titleAr', 'themeLabelAr', 'tasks', 'newWords'],
          },
        },
      });

      const parsed = JSON.parse((response.text ?? '{}').trim()) as {
        titleAr?: string;
        themeLabelAr?: string;
        tasks?: Array<Record<string, unknown>>;
        newWords?: Array<Record<string, unknown>>;
      };

      const dailyPayload = {
        dateKey: todayStr,
        titleAr: parsed.titleAr || 'تحدي اليوم السريع',
        themeLabelAr: parsed.themeLabelAr || 'مفردات مختارة',
        tasks: (parsed.tasks ?? []).map((t, idx) => ({
          id: String(t.id || `task-${idx + 1}`),
          type: ['choice', 'fill_blank', 'speak'].includes(String(t.type))
            ? String(t.type)
            : 'choice',
          promptAr: String(t.promptAr || ''),
          promptEn: String(t.promptEn || ''),
          options: Array.isArray(t.options) ? t.options.map(String) : undefined,
          correctAnswer: String(t.correctAnswer || ''),
          tipAr: String(t.tipAr || ''),
        })),
        newWords: (parsed.newWords ?? []).map((w) => ({
          word: String(w.word || ''),
          arabicMeaning: String(w.arabicMeaning || ''),
          partOfSpeech: ['noun', 'verb', 'adjective', 'adverb', 'phrase', 'other'].includes(
            String(w.partOfSpeech).toLowerCase()
          )
            ? String(w.partOfSpeech).toLowerCase()
            : 'other',
          phonetic: String(w.phonetic || ''),
          exampleEn: String(w.exampleEn || ''),
          exampleAr: String(w.exampleAr || ''),
          level: ['beginner', 'intermediate', 'advanced'].includes(
            String(w.level).toLowerCase()
          )
            ? String(w.level).toLowerCase()
            : 'intermediate',
        })),
      };

      if (dailyCacheKey) {
        setBoundedCache(dailyChallengeCache, dailyCacheKey, dailyPayload, MAX_DAILY_CACHE);
      }

      res.json(dailyPayload);
    } catch (error) {
      console.error('Error in /api/daily-challenge/generate:', error);
      const classified = classifyGeminiHttpError(error);
      if (classified.isRateLimit) {
        res.status(429).json({
          error: 'تم تجاوز حد الطلبات المؤقت المجاني. يرجى المحاولة بعد لحظات.',
        });
        return;
      }
      res.status(500).json({
        error: 'تعذر تجهيز تحدي اليوم حالياً. يرجى المحاولة بعد قليل.',
      });
    }
  });

  // 5b. Speech-to-Text for Voice Conversation with the AI Tutor (gemini-3.5-transcribe)
  app.post('/api/chat/transcribe', async (req, res) => {
    try {
      const { audioBase64, mimeType } = (req.body || {}) as {
        audioBase64?: string;
        mimeType?: string;
      };

      if (!audioBase64 || typeof audioBase64 !== 'string' || audioBase64.length > 12_000_000) {
        res.status(400).json({ error: 'يرجى تسجيل رسالة صوتية صالحة أولاً.' });
        return;
      }

      const cleanMime = (mimeType || 'audio/webm').split(';')[0].trim();
      const ai = getAiClient();

      const transcribeRes = await ai.models.generateContent({
        model: 'gemini-3.5-transcribe',
        contents: [
          {
            role: 'user',
            parts: [
              {
                inlineData: {
                  mimeType: cleanMime,
                  data: audioBase64,
                },
              },
              {
                text: 'Transcribe the spoken audio accurately. The speaker is an Arabic learner practicing English and may speak in English, Arabic, or a mix of both. Return ONLY the exact spoken words without explanations, or "(silence)" if no speech is audible.',
              },
            ],
          },
        ],
      });

      const transcript = (transcribeRes.text ?? '').trim();
      if (!transcript || transcript.toLowerCase() === '(silence)' || transcript.toLowerCase() === 'silence') {
        res.json({ transcript: '' });
        return;
      }

      res.json({ transcript });
    } catch (error) {
      console.error('Error in POST /api/chat/transcribe:', error);
      const classified = classifyGeminiHttpError(error);
      res.status(classified.status).json({
        error: classified.isRateLimit
          ? 'تم تجاوز حد الطلبات المؤقت. يرجى الانتظار بضع ثوانٍ والمحاولة مجدداً.'
          : 'تعذر تحويل الصوت إلى نص حالياً.',
      });
    }
  });

  // Evaluate user-composed English sentence using 2-3 saved vocabulary words and suggest natural American phrasing
  app.post('/api/evaluate-user-sentence', async (req, res) => {
    try {
      const body = (req.body || {}) as {
        targetWords?: Array<{ word?: string; arabicMeaning?: string; partOfSpeech?: string }>;
        userSentence?: string;
      };

      const rawSentence = typeof body.userSentence === 'string' ? body.userSentence.trim() : '';
      if (rawSentence.length < 3) {
        res.status(400).json({ error: 'يرجى كتابة جملة إنجليزية كاملة أولاً.' });
        return;
      }
      if (rawSentence.length > 420) {
        res.status(400).json({ error: 'الجملة طويلة جداً (الحد الأقصى 420 حرفاً).' });
        return;
      }

      const targetWords = (Array.isArray(body.targetWords) ? body.targetWords : [])
        .map((item) => ({
          word: typeof item?.word === 'string' ? item.word.trim().slice(0, 60) : '',
          arabicMeaning:
            typeof item?.arabicMeaning === 'string' ? item.arabicMeaning.trim().slice(0, 120) : '',
          partOfSpeech:
            typeof item?.partOfSpeech === 'string' ? item.partOfSpeech.trim().slice(0, 30) : '',
        }))
        .filter((item) => item.word.length > 0)
        .slice(0, 4);

      if (targetWords.length === 0) {
        res.status(400).json({ error: 'يرجى تحديد الكلمات المستهدفة للتقييم.' });
        return;
      }

      const wordsListDesc = targetWords
        .map(
          (w, idx) =>
            `${idx + 1}. "${w.word}"${w.arabicMeaning ? ` (${w.arabicMeaning})` : ''}${w.partOfSpeech ? ` [${w.partOfSpeech}]` : ''}`
        )
        .join('\n');

      const ai = getAiClient();
      const prompt = `You are a supportive American English tutor evaluating an Arabic-speaking learner's sentence.

Target vocabulary words the learner was asked to use together in ONE coherent English sentence:
${wordsListDesc}

Learner's written sentence:
"${rawSentence}"

Evaluate carefully and return ONLY valid JSON:
1. Check if each target word (or a valid grammatical inflection like plural/tense) appears meaningfully and accurately in the sentence.
2. Evaluate grammar, prepositions, word order, spelling, and naturalness (score 0 to 100; isValid true if grammatically understandable and uses the target words properly).
3. In "grammarIssuesAr", write clear, helpful feedback in Arabic: praise what they did well, point out any grammar/spelling/usage errors or missing target words.
4. In "naturalAmericanSentence", write ONE smooth, authentic, everyday **American English** sentence that naturally incorporates ALL of the target words together (preserving the learner's idea if close, or polishing it into idiomatic US English).
5. In "naturalAmericanAr", provide an accurate natural Arabic translation of "naturalAmericanSentence".
6. In "tipAr", give 1 brief practical Arabic tip on why the American phrasing sounds natural.`;

      const response = await withTimeout(
        generateWithFreeFallback(ai, {
          contents: prompt,
          config: {
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            responseMimeType: 'application/json',
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                isValid: { type: Type.BOOLEAN },
                score: { type: Type.INTEGER },
                usedWords: { type: Type.ARRAY, items: { type: Type.STRING } },
                missingWords: { type: Type.ARRAY, items: { type: Type.STRING } },
                grammarIssuesAr: { type: Type.STRING },
                naturalAmericanSentence: { type: Type.STRING },
                naturalAmericanAr: { type: Type.STRING },
                tipAr: { type: Type.STRING },
              },
              required: [
                'isValid',
                'score',
                'usedWords',
                'missingWords',
                'grammarIssuesAr',
                'naturalAmericanSentence',
                'naturalAmericanAr',
                'tipAr',
              ],
            },
          },
        }),
        12000,
        'evaluate-user-sentence'
      );

      const rawText = (response.text || '{}')
        .trim()
        .replace(/^```json\s*/i, '')
        .replace(/^```\s*/i, '')
        .replace(/\s*```$/i, '');
      const parsed = JSON.parse(rawText) as {
        isValid?: boolean;
        score?: number;
        usedWords?: string[];
        missingWords?: string[];
        grammarIssuesAr?: string;
        naturalAmericanSentence?: string;
        naturalAmericanAr?: string;
        tipAr?: string;
      };

      const score =
        typeof parsed.score === 'number' && Number.isFinite(parsed.score)
          ? Math.max(0, Math.min(100, Math.round(parsed.score)))
          : 70;

      res.json({
        isValid: Boolean(parsed.isValid),
        score,
        usedWords: Array.isArray(parsed.usedWords) ? parsed.usedWords : [],
        missingWords: Array.isArray(parsed.missingWords) ? parsed.missingWords : [],
        grammarIssuesAr:
          typeof parsed.grammarIssuesAr === 'string' && parsed.grammarIssuesAr.trim()
            ? parsed.grammarIssuesAr.trim()
            : 'جملة جيدة، راجع الصياغة الأمريكية المقترحة أدناه.',
        naturalAmericanSentence:
          typeof parsed.naturalAmericanSentence === 'string' && parsed.naturalAmericanSentence.trim()
            ? parsed.naturalAmericanSentence.trim()
            : rawSentence,
        naturalAmericanAr:
          typeof parsed.naturalAmericanAr === 'string' ? parsed.naturalAmericanAr.trim() : '',
        tipAr: typeof parsed.tipAr === 'string' ? parsed.tipAr.trim() : '',
      });
    } catch (error) {
      console.error('Error in POST /api/evaluate-user-sentence:', error);
      const classified = classifyGeminiHttpError(error);
      if (classified.isRateLimit) {
        res.status(429).json({
          error: 'الخادم مزدحم مؤقتاً بطلبات المعلم الذكي. يرجى الانتظار بضع ثوانٍ والمحاولة مجدداً.',
        });
        return;
      }
      res.status(500).json({
        error: 'تعذر تقييم الجملة عبر المعلم الذكي حالياً. تأكد من توفر الاتصال وحاول مجدداً.',
      });
    }
  });

  // 6. Smart Bilingual & Voice-Ready AI English Tutor Chat (Streaming SSE via gemini-4-hydrogen)
  app.post('/api/chat', async (req, res) => {
    try {
      const {
        messages = [],
        chatMode = 'conversation',
        userLevel = null,
        userWords = [],
        userWeakWords = [],
        voiceConversation = false,
        dialect = 'us',
        scenario = 'free',
      } = (req.body || {}) as {
        messages?: Array<{ role: 'user' | 'model'; text: string }>;
        chatMode?: 'conversation' | 'grammar' | 'vocab';
        userLevel?: string | null;
        userWords?: Array<{ word: string; arabicMeaning: string }>;
        userWeakWords?: Array<{ word: string; arabicMeaning: string }>;
        voiceConversation?: boolean;
        dialect?: 'us' | 'uk';
        scenario?: string;
      };

      const cleanMessages = messages
        .filter(
          (m) =>
            m &&
            (m.role === 'user' || m.role === 'model') &&
            typeof m.text === 'string' &&
            m.text.trim().length > 0
        )
        .slice(-20)
        .map((m) => ({
          role: m.role,
          parts: [{ text: m.text.trim().slice(0, 2000) }],
        }));

      if (cleanMessages.length === 0) {
        res.status(400).json({ error: 'يرجى كتابة رسالة أو التحدث بالميكروفون للبدء.' });
        return;
      }

      const vocabSummary = userWords
        .slice(0, 12)
        .map((w) => `${w.word} (${w.arabicMeaning})`)
        .join(', ');

      const weakWordsSummary = userWeakWords
        .slice(0, 8)
        .map((w) => `${w.word} (${w.arabicMeaning})`)
        .join(', ');

      const levelHint =
        userLevel === 'beginner'
          ? 'The learner level is Beginner (مبتدئ) — keep English sentences simple, short, and clear.'
          : userLevel === 'advanced'
            ? 'The learner level is Advanced (متقدم) — use rich natural idiomatic English.'
            : 'The learner level is Intermediate (متوسط) — use natural everyday English.';

      const dialectHint =
        dialect === 'uk'
          ? 'Use natural British English phrasing and spelling.'
          : 'Use natural American English phrasing and spelling.';

      const scenarioLabels: Record<string, string> = {
        free: 'Free everyday English conversation',
        travel: 'Airport, hotel, and travel roleplay',
        work: 'Workplace communication and job interview roleplay',
        restaurant: 'Ordering food at a café or restaurant roleplay',
        daily: 'Daily life, shopping, and making friends',
        weak_words: `Actively practicing the learner's challenging words (${weakWordsSummary || vocabSummary || 'saved words'})`,
      };
      const scenarioDesc = scenarioLabels[scenario] || scenarioLabels.free;

      const voiceFormattingGuide = voiceConversation
        ? 'IMPORTANT FOR VOICE CONVERSATION: Write your response in clean, natural spoken language without heavy markdown symbols (#, *, bullet lists) so it sounds natural when read aloud. First give your spoken English response (2-3 crisp conversational sentences ending with a natural question). Then if the user made a mistake or needs clarity, add a brief line starting with "💡 تصحيح:" or "الترجمة:" in short Arabic.'
        : 'Format clearly without clutter. If the user made any English mistake, include a gentle "💡 التصحيح:" line with a short Arabic explanation, plus an engaging follow-up question or practical next step.';

      const modeInstruction =
        chatMode === 'grammar'
          ? `Mode: Grammar & Practical Usage Coach. Explain concisely in Arabic, give 2-3 clear everyday English examples with Arabic meanings, point out any common trap to avoid, and invite the learner to try 1 practice sentence.`
          : chatMode === 'vocab'
            ? `Mode: Personal Vocabulary Trainer. Actively help the learner practice their saved words (${vocabSummary || 'common English words'})${weakWordsSummary ? ` and prioritize weak/mistake words (${weakWordsSummary})` : ''}. Use short interactive dialogues, fill-in-the-blank challenges, or ask them to reply using a target word.`
            : `Mode: Bilingual English Conversation Partner. Scenario: ${scenarioDesc}. Reply warmly in natural English (2-3 sentences) to keep a real dialogue going, subtly weave in saved/weak words when natural (${weakWordsSummary || vocabSummary || 'everyday words'}), gently correct mistakes in Arabic if any, and end with an engaging English question.`;

      const systemInstruction = `You are a highly effective, supportive bilingual English-Arabic personal tutor inside the "English Card" app.
${levelHint}
${dialectHint}
${modeInstruction}
${voiceFormattingGuide}`;

      const ai = getAiClient();

      res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');

      const stream = await generateStreamWithFreeFallback(ai, {
        contents: cleanMessages,
        config: {
          systemInstruction,
          thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        },
      });

      for await (const chunk of stream) {
        const text = chunk.text;
        if (text) {
          res.write(`data: ${JSON.stringify({ text })}\n\n`);
        }
      }

      res.write('data: [DONE]\n\n');
      res.end();
    } catch (error) {
      console.error('Error in /api/chat:', error);
      const classified = classifyGeminiHttpError(error);
      const errMessage = classified.isRateLimit
        ? 'تم تجاوز حد الطلبات المؤقت المجاني. يرجى الانتظار بضع ثوانٍ ثم الإرسال مجدداً.'
        : 'تعذر الاتصال بالمعلم الذكي حالياً. حاول مجدداً.';
      if (!res.headersSent) {
        res.status(classified.status).json({ error: errMessage });
      } else {
        res.write(`data: ${JSON.stringify({ error: errMessage })}\n\n`);
        res.end();
      }
    }
  });

  // Built-in Arabic-English Dictionary File, Professional Specialties & App-Path Export Storage
  app.get('/api/dictionary', (req, res) => {
    const q = typeof req.query.q === 'string' ? req.query.q : '';
    const category = typeof req.query.category === 'string' ? req.query.category : 'all';
    const level = typeof req.query.level === 'string' ? req.query.level : 'all';
    const specialtyParam = typeof req.query.specialty === 'string' ? req.query.specialty : 'general';
    const specialty = normalizeSpecialty(specialtyParam);

    const generalEntries = getBuiltInDictionaryEntries();
    const specialtyEntries = specialty !== 'general' ? getSpecialtyEntries(specialty) : [];

    const sourceEntries =
      specialty !== 'general'
        ? specialtyEntries
        : generalEntries;

    const qLower = q.trim().toLowerCase();
    const filtered =
      specialty !== 'general'
        ? sourceEntries.filter((entry) => {
            if (category && category !== 'all' && entry.category !== category) return false;
            if (level && level !== 'all' && entry.level !== level) return false;
            if (!qLower) return true;
            return (
              entry.word.toLowerCase().includes(qLower) ||
              entry.arabicMeaning.toLowerCase().includes(qLower) ||
              entry.exampleEn.toLowerCase().includes(qLower) ||
              entry.exampleAr.toLowerCase().includes(qLower)
            );
          })
        : searchBuiltInDictionary({ query: q, category, level });

    const categories = Array.from(new Set(sourceEntries.map((e) => e.category)));

    res.json({
      entries: filtered,
      categories,
      totalCount: sourceEntries.length,
      specialty,
    });
  });

  app.post('/api/dictionary/generate-specialty', async (req, res) => {
    try {
      const body = (req.body || {}) as {
        specialty?: string;
        focusTopic?: string;
        level?: string;
        count?: number;
      };
      const specialty = normalizeSpecialty(body.specialty);
      const specialtyMeta = getSpecialtyMeta(specialty);
      const focusTopic =
        typeof body.focusTopic === 'string' ? body.focusTopic.trim().slice(0, 120) : '';
      const requestedLevel =
        body.level === 'beginner' || body.level === 'intermediate' || body.level === 'advanced'
          ? body.level
          : 'all';
      const count = Math.min(12, Math.max(6, Number(body.count) || 8));

      const fallbackEntries =
        specialty !== 'general'
          ? getSpecialtyEntries(specialty)
          : getBuiltInDictionaryEntries().slice(0, count);

      if (hasConfiguredGeminiKey()) {
        try {
          const ai = getAiClient();
          const levelRule =
            requestedLevel === 'all'
              ? 'Include a practical mix of beginner, intermediate, and advanced workplace terms.'
              : `Target difficulty level: ${requestedLevel}.`;
          const focusRule = focusTopic
            ? `Specifically emphasize the sub-topic or role: "${focusTopic}".`
            : '';

          const aiRes = await withTimeout(
            generateWithFreeFallback(ai, {
              contents: `Generate ${count} high-value English vocabulary words or short professional expressions tailored for an Arabic-speaking learner working or studying in the field of: ${specialtyMeta.englishDomain} (${specialtyMeta.label}).
${levelRule}
${focusRule}

Rules:
- Every word must be genuinely useful in ${specialtyMeta.englishDomain}.
- Each "exampleEn" must be a realistic, authentic sentence from that exact profession/workplace, with an accurate Arabic translation in "exampleAr".
- Provide accurate US/UK IPA phonetics in "phonetic", a clear Arabic meaning in "arabicMeaning", and rich "usageContext" (common synonym, professional collocation + Arabic meaning, and a concise Arabic explanation of when professionals say this term).`,
              config: {
                thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
                responseMimeType: 'application/json',
                responseSchema: {
                  type: Type.OBJECT,
                  properties: {
                    entries: {
                      type: Type.ARRAY,
                      items: {
                        type: Type.OBJECT,
                        properties: {
                          word: { type: Type.STRING },
                          arabicMeaning: { type: Type.STRING },
                          partOfSpeech: { type: Type.STRING },
                          phonetic: { type: Type.STRING },
                          exampleEn: { type: Type.STRING },
                          exampleAr: { type: Type.STRING },
                          level: { type: Type.STRING },
                          synonym: { type: Type.STRING },
                          collocation: { type: Type.STRING },
                          collocationAr: { type: Type.STRING },
                          dailySituationAr: { type: Type.STRING },
                        },
                        required: [
                          'word',
                          'arabicMeaning',
                          'partOfSpeech',
                          'phonetic',
                          'exampleEn',
                          'exampleAr',
                          'level',
                        ],
                      },
                    },
                  },
                  required: ['entries'],
                },
              },
            }),
            9000,
            'dictionary-generate-specialty'
          );

          const rawText = (aiRes.text ?? '{}')
            .trim()
            .replace(/^```json\s*/i, '')
            .replace(/^```\s*/i, '')
            .replace(/\s*```$/i, '');
          const parsed = JSON.parse(rawText) as {
            entries?: Array<{
              word?: string;
              arabicMeaning?: string;
              partOfSpeech?: string;
              phonetic?: string;
              exampleEn?: string;
              exampleAr?: string;
              level?: string;
              synonym?: string;
              collocation?: string;
              collocationAr?: string;
              dailySituationAr?: string;
            }>;
          };

          const seen = new Set<string>();
          const generated: Array<
            BilingualDictionaryEntry & {
              specialty: string;
              usageContext?: {
                synonym: string;
                collocation: string;
                collocationAr: string;
                dailySituationAr: string;
              };
            }
          > = [];

          for (const item of parsed.entries ?? []) {
            const word = String(item.word || '').trim();
            const arabicMeaning = String(item.arabicMeaning || '').trim();
            if (!word || !arabicMeaning) continue;
            const lower = word.toLowerCase();
            if (seen.has(lower)) continue;
            seen.add(lower);

            const posRaw = String(item.partOfSpeech || 'other').toLowerCase();
            const partOfSpeech: BilingualDictionaryEntry['partOfSpeech'] =
              posRaw === 'noun' ||
              posRaw === 'verb' ||
              posRaw === 'adjective' ||
              posRaw === 'adverb' ||
              posRaw === 'phrase'
                ? posRaw
                : 'other';

            const lvlRaw = String(item.level || 'intermediate').toLowerCase();
            const level: BilingualDictionaryEntry['level'] =
              lvlRaw === 'beginner' || lvlRaw === 'intermediate' || lvlRaw === 'advanced'
                ? lvlRaw
                : 'intermediate';

            const baseIpa = String(item.phonetic || '').trim() || `/${lower}/`;
            const dialect = buildDualDialectInfo(word, baseIpa);

            generated.push({
              word,
              arabicMeaning,
              partOfSpeech,
              phonetic: dialect.phoneticUs,
              phoneticUs: dialect.phoneticUs,
              phoneticUk: dialect.phoneticUk,
              ...(dialect.spellingUk ? { spellingUk: dialect.spellingUk } : {}),
              exampleEn: String(item.exampleEn || '').trim(),
              exampleAr: String(item.exampleAr || '').trim(),
              level,
              category: focusTopic ? `${specialtyMeta.label} — ${focusTopic}` : specialtyMeta.label,
              specialty,
              usageContext: {
                synonym: String(item.synonym || word).trim(),
                collocation: String(item.collocation || word).trim(),
                collocationAr: String(item.collocationAr || arabicMeaning).trim(),
                dailySituationAr: String(item.dailySituationAr || specialtyMeta.descriptionAr).trim(),
              },
            });
          }

          if (generated.length > 0) {
            res.json({
              entries: generated,
              specialty,
              specialtyLabel: specialtyMeta.label,
              mode: 'ai',
            });
            return;
          }
        } catch {
          // Fall back cleanly to curated professional entries below
        }
      }

      const filteredFallback =
        requestedLevel === 'all'
          ? fallbackEntries
          : fallbackEntries.filter((e) => e.level === requestedLevel);

      res.json({
        entries: filteredFallback.length > 0 ? filteredFallback : fallbackEntries,
        specialty,
        specialtyLabel: specialtyMeta.label,
        mode: 'curated',
      });
    } catch (error) {
      console.error('Error in POST /api/dictionary/generate-specialty:', error);
      res.status(500).json({ error: 'تعذر توليد مفردات التخصص حالياً. حاول مجدداً.' });
    }
  });

  // Save dictionary file inside app path (`data/dictionaries/`) and return download info
  app.post('/api/dictionary/save-export', (req, res) => {
    try {
      const body = (req.body || {}) as {
        format?: string;
        specialty?: string;
        customSlug?: string;
        entries?: Array<Partial<BilingualDictionaryEntry>>;
      };
      const format: 'csv' | 'json' =
        String(body.format || 'csv').toLowerCase() === 'json' ? 'json' : 'csv';
      const specialty = normalizeSpecialty(body.specialty);
      const specialtyMeta = getSpecialtyMeta(specialty);

      let entriesToSave: BilingualDictionaryEntry[] = [];
      if (Array.isArray(body.entries) && body.entries.length > 0) {
        entriesToSave = body.entries
          .filter((e) => e && typeof e.word === 'string' && e.word.trim() && typeof e.arabicMeaning === 'string')
          .slice(0, 500)
          .map((e) => {
            const word = String(e.word || '').trim();
            const dialect = buildDualDialectInfo(word, String(e.phonetic || ''));
            const posRaw = String(e.partOfSpeech || 'other').toLowerCase();
            const lvlRaw = String(e.level || 'intermediate').toLowerCase();
            return {
              word,
              arabicMeaning: String(e.arabicMeaning || '').trim(),
              partOfSpeech:
                posRaw === 'noun' ||
                posRaw === 'verb' ||
                posRaw === 'adjective' ||
                posRaw === 'adverb' ||
                posRaw === 'phrase'
                  ? posRaw
                  : 'other',
              phonetic: String(e.phonetic || dialect.phoneticUs),
              phoneticUs: String(e.phoneticUs || dialect.phoneticUs),
              phoneticUk: String(e.phoneticUk || dialect.phoneticUk),
              ...(e.spellingUk || dialect.spellingUk
                ? { spellingUk: String(e.spellingUk || dialect.spellingUk) }
                : {}),
              exampleEn: String(e.exampleEn || '').trim(),
              exampleAr: String(e.exampleAr || '').trim(),
              level:
                lvlRaw === 'beginner' || lvlRaw === 'intermediate' || lvlRaw === 'advanced'
                  ? lvlRaw
                  : 'intermediate',
              category: String(e.category || specialtyMeta.label).trim(),
            };
          });
      }

      if (entriesToSave.length === 0) {
        entriesToSave =
          specialty !== 'general'
            ? getSpecialtyEntries(specialty)
            : getBuiltInDictionaryEntries();
      }

      const { record } = saveDictionaryFileToAppPath({
        entries: entriesToSave,
        format,
        specialty,
        customSlug: body.customSlug,
      });

      res.status(201).json({
        file: record,
        downloadUrl: `/api/dictionary/saved/${encodeURIComponent(record.fileName)}/download`,
      });
    } catch (error) {
      console.error('Error in POST /api/dictionary/save-export:', error);
      res.status(500).json({ error: 'تعذر حفظ القاموس داخل مسار التطبيق.' });
    }
  });

  app.get('/api/dictionary/saved', (_req, res) => {
    try {
      res.json({
        directoryPath: 'data/dictionaries',
        files: listSavedDictionaryFiles(),
      });
    } catch (error) {
      console.error('Error in GET /api/dictionary/saved:', error);
      res.status(500).json({ error: 'تعذر قراءة القواميس المحفوظة في مسار التطبيق.' });
    }
  });

  app.get('/api/dictionary/saved/:fileName', (req, res) => {
    try {
      const loaded = readSavedDictionaryEntries(String(req.params.fileName || ''));
      if (!loaded) {
        res.status(404).json({ error: 'لم يتم العثور على ملف القاموس في مسار التطبيق.' });
        return;
      }
      res.json({
        file: loaded.record,
        entries: loaded.entries,
      });
    } catch (error) {
      console.error('Error in GET /api/dictionary/saved/:fileName:', error);
      res.status(500).json({ error: 'تعذر فتح ملف القاموس المحفوظ.' });
    }
  });

  app.get('/api/dictionary/saved/:fileName/download', (req, res) => {
    try {
      const fileName = String(req.params.fileName || '');
      const target = getSafeDictionaryFilePath(fileName);
      if (!target || !fs.existsSync(target)) {
        res.status(404).json({ error: 'ملف القاموس غير موجود.' });
        return;
      }
      const isJson = fileName.endsWith('.json');
      res.setHeader(
        'Content-Type',
        isJson ? 'application/json; charset=utf-8' : 'text/csv; charset=utf-8'
      );
      res.setHeader('Content-Disposition', `attachment; filename="${fileName}"`);
      res.send(fs.readFileSync(target));
    } catch (error) {
      console.error('Error in GET /api/dictionary/saved/:fileName/download:', error);
      res.status(500).json({ error: 'تعذر تنزيل ملف القاموس.' });
    }
  });

  app.delete('/api/dictionary/saved/:fileName', (req, res) => {
    try {
      const deleted = deleteSavedDictionaryFile(String(req.params.fileName || ''));
      if (!deleted) {
        res.status(404).json({ error: 'ملف القاموس غير موجود.' });
        return;
      }
      res.json({ ok: true });
    } catch (error) {
      console.error('Error in DELETE /api/dictionary/saved/:fileName:', error);
      res.status(500).json({ error: 'تعذر حذف ملف القاموس.' });
    }
  });

  app.get('/api/dictionary/download', (req, res) => {
    const format: 'csv' | 'json' =
      String(req.query.format || 'csv').toLowerCase() === 'json' ? 'json' : 'csv';
    const specialty = normalizeSpecialty(req.query.specialty);
    const entries =
      specialty !== 'general' ? getSpecialtyEntries(specialty) : getBuiltInDictionaryEntries();

    // Always persist a copy into the app path (`data/dictionaries/`) on download
    let savedFileName = `arabic-english-dictionary.${format}`;
    let content = '';
    try {
      const saved = saveDictionaryFileToAppPath({ entries, format, specialty });
      savedFileName = saved.record.fileName;
      content = saved.content;
    } catch {
      content =
        format === 'json'
          ? JSON.stringify(entries, null, 2)
          : buildDictionaryCsvString(entries);
    }

    if (format === 'json') {
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      res.setHeader('Content-Disposition', `attachment; filename="${savedFileName}"`);
      res.send(content);
      return;
    }

    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename="${savedFileName}"`);
    res.send(content);
  });

  app.post('/api/translate', async (req, res) => {
    try {
      const body = (req.body || {}) as {
        text?: string;
        direction?: 'auto' | 'en-ar' | 'ar-en';
        specialty?: string;
      };

      const cleanText = typeof body.text === 'string' ? body.text.trim().slice(0, 800) : '';
      if (!cleanText) {
        res.status(400).json({ error: 'يرجى كتابة كلمة أو جملة للترجمة.' });
        return;
      }

      const direction =
        body.direction === 'en-ar' || body.direction === 'ar-en' ? body.direction : 'auto';
      const specialty = normalizeSpecialty(body.specialty);
      const specialtyMeta = getSpecialtyMeta(specialty);

      // Check curated specialty entries first when a professional domain is selected
      if (specialty !== 'general') {
        const specialtyHit = getSpecialtyEntries(specialty).find(
          (e) =>
            e.word.toLowerCase() === cleanText.toLowerCase() ||
            e.arabicMeaning.includes(cleanText)
        );
        if (specialtyHit) {
          const hasArabicInput = /[\u0600-\u06FF]/.test(cleanText);
          const srcLang: 'ar' | 'en' =
            direction === 'ar-en' || (direction === 'auto' && hasArabicInput) ? 'ar' : 'en';
          const tgtLang: 'ar' | 'en' = srcLang === 'ar' ? 'en' : 'ar';
          res.json({
            sourceText: cleanText,
            translatedText: srcLang === 'en' ? specialtyHit.arabicMeaning : specialtyHit.word,
            sourceLang: srcLang,
            targetLang: tgtLang,
            matchedEntry: specialtyHit,
            relatedMatches: [],
            suggestedCard: specialtyHit,
            translationMode: `specialty-${specialty}`,
          });
          return;
        }
      }

      // Fast path: check built-in bilingual dictionary & free key-less translator
      const freeResult = await translateBilingualFree({ text: cleanText, direction });
      if (freeResult.matchedEntry && specialty === 'general') {
        res.json(freeResult);
        return;
      }

      // Rich domain-tailored translation & context generation via Gemini when configured
      if (
        hasConfiguredGeminiKey() &&
        (specialty !== 'general' || cleanText.split(/\s+/).length >= 2)
      ) {
        try {
          const ai = getAiClient();
          const hasArabic = /[\u0600-\u06FF]/.test(cleanText);
          const srcLang: 'ar' | 'en' =
            direction === 'ar-en' || (direction === 'auto' && hasArabic) ? 'ar' : 'en';
          const tgtLang: 'ar' | 'en' = srcLang === 'ar' ? 'en' : 'ar';
          const domainPrompt =
            specialty !== 'general'
              ? `Tailor the translation nuance, "exampleEn", "exampleAr", and "usageContext" (synonym, collocation, collocationAr, dailySituationAr) specifically to the professional field of ${specialtyMeta.englishDomain} (${specialtyMeta.label}).`
              : `Return a clear, practical everyday English example sentence and Arabic translation.`;

          const aiRes = await withTimeout(
            generateWithFreeFallback(ai, {
              contents: `Translate this ${srcLang === 'ar' ? 'Arabic' : 'English'} text naturally into ${tgtLang === 'en' ? 'English' : 'Arabic'} for a learner, and return one flashcard item for the core word or phrase.
${domainPrompt}

Input:
"""
${cleanText}
"""`,
              config: {
                thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
                responseMimeType: 'application/json',
                responseSchema: {
                  type: Type.OBJECT,
                  properties: {
                    translatedText: { type: Type.STRING },
                    keyWordEn: { type: Type.STRING },
                    keyMeaningAr: { type: Type.STRING },
                    phonetic: { type: Type.STRING },
                    partOfSpeech: { type: Type.STRING },
                    exampleEn: { type: Type.STRING },
                    exampleAr: { type: Type.STRING },
                    level: { type: Type.STRING },
                    synonym: { type: Type.STRING },
                    collocation: { type: Type.STRING },
                    collocationAr: { type: Type.STRING },
                    dailySituationAr: { type: Type.STRING },
                  },
                  required: ['translatedText'],
                },
              },
            }),
            5500,
            'translate'
          );

          const parsed = JSON.parse((aiRes.text ?? '{}').trim()) as {
            translatedText?: string;
            keyWordEn?: string;
            keyMeaningAr?: string;
            phonetic?: string;
            partOfSpeech?: string;
            exampleEn?: string;
            exampleAr?: string;
            level?: string;
            synonym?: string;
            collocation?: string;
            collocationAr?: string;
            dailySituationAr?: string;
          };

          if (parsed.translatedText && parsed.translatedText.trim()) {
            const wordEn = (parsed.keyWordEn || (tgtLang === 'en' ? parsed.translatedText : cleanText)).trim();
            const meaningAr = (parsed.keyMeaningAr || (tgtLang === 'ar' ? parsed.translatedText : cleanText)).trim();
            const baseIpa = (parsed.phonetic || '').trim() || `/${wordEn.toLowerCase()}/`;
            const dialect = wordEn ? buildDualDialectInfo(wordEn, baseIpa) : undefined;
            res.json({
              sourceText: cleanText,
              translatedText: parsed.translatedText.trim(),
              sourceLang: srcLang,
              targetLang: tgtLang,
              relatedMatches: freeResult.relatedMatches,
              suggestedCard: wordEn
                ? {
                    word: wordEn,
                    arabicMeaning: meaningAr,
                    partOfSpeech: ['noun', 'verb', 'adjective', 'adverb', 'phrase', 'other'].includes(
                      String(parsed.partOfSpeech || '').toLowerCase()
                    )
                      ? String(parsed.partOfSpeech).toLowerCase()
                      : 'phrase',
                    phonetic: dialect?.phoneticUs || baseIpa,
                    phoneticUs: dialect?.phoneticUs,
                    phoneticUk: dialect?.phoneticUk,
                    spellingUk: dialect?.spellingUk,
                    exampleEn: (parsed.exampleEn || '').trim(),
                    exampleAr: (parsed.exampleAr || '').trim(),
                    level: ['beginner', 'intermediate', 'advanced'].includes(
                      String(parsed.level || '').toLowerCase()
                    )
                      ? String(parsed.level).toLowerCase()
                      : 'intermediate',
                    ...(parsed.synonym || parsed.collocation || parsed.dailySituationAr
                      ? {
                          usageContext: {
                            synonym: (parsed.synonym || wordEn).trim(),
                            collocation: (parsed.collocation || wordEn).trim(),
                            collocationAr: (parsed.collocationAr || meaningAr).trim(),
                            dailySituationAr: (
                              parsed.dailySituationAr || specialtyMeta.descriptionAr
                            ).trim(),
                          },
                        }
                      : {}),
                  }
                : freeResult.suggestedCard,
              translationMode: specialty !== 'general' ? `gemini-${specialty}` : 'gemini-assisted',
            });
            return;
          }
        } catch {
          // Fall back seamlessly to freeResult below
        }
      }

      if (specialty !== 'general' && freeResult.suggestedCard) {
        const adapted = adaptCardContextForSpecialty(
          freeResult.suggestedCard.word,
          freeResult.suggestedCard.arabicMeaning,
          specialty
        );
        res.json({
          ...freeResult,
          suggestedCard: {
            ...freeResult.suggestedCard,
            exampleEn: adapted.exampleEn,
            exampleAr: adapted.exampleAr,
            usageContext: adapted.usageContext,
          },
        });
        return;
      }

      res.json(freeResult);
    } catch (error) {
      console.error('Error in POST /api/translate:', error);
      res.status(500).json({ error: 'تعذر إتمام الترجمة حالياً. حاول مجدداً.' });
    }
  });

  // Public aggregate application stats (subscribers count only, zero PII)
  app.get('/api/stats', (_req, res) => {
    res.json({ subscribersCount: getSubscribersCount() });
  });

  // Community Rooms (Real-time user-to-user chat & voice rooms)
  app.get('/api/community/rooms', (_req, res) => {
    res.json({ rooms: listPublicCommunityRooms() });
  });

  app.get('/api/community/rooms/:roomId', (req, res) => {
    const room = getCommunityRoomSummary(req.params.roomId || '');
    if (!room) {
      res.status(404).json({ error: 'لم يتم العثور على الغرفة.' });
      return;
    }
    res.json({ room });
  });

  app.post('/api/community/rooms', (req, res) => {
    try {
      const auth = getAuthenticatedSession(req);
      if (auth?.viaCookie && !verifySameOriginOrHeader(req)) {
        res.status(403).json({ error: 'طلب غير مصرح به.' });
        return;
      }
      const authUser = auth ? findUserById(auth.session.uid) : null;
      const body = (req.body || {}) as {
        title?: string;
        isPrivate?: boolean;
        guestName?: string;
      };

      const creatorLabel = authUser
        ? authUser.username
        : typeof body.guestName === 'string' && body.guestName.trim()
          ? body.guestName.trim().slice(0, 24)
          : 'مستخدم';

      const result = createCommunityRoom({
        title: String(body.title ?? ''),
        isPrivate: Boolean(body.isPrivate),
        createdBy: creatorLabel,
      });

      if (!result.ok) {
        const fail = result as { ok: false; error: string };
        res.status(400).json({ error: fail.error });
        return;
      }

      res.status(201).json({ room: result.room });
    } catch (error) {
      console.error('Error in POST /api/community/rooms:', error);
      res.status(500).json({ error: 'تعذر إنشاء غرفة التواصل حالياً.' });
    }
  });

  const distPath = path.resolve(__dirname, 'dist');
  const distIndexPath = path.join(distPath, 'index.html');
  let useViteDevMiddleware = process.env.NODE_ENV !== 'production';

  if (useViteDevMiddleware) {
    try {
      const { createServer: createViteServer } = await import('vite');
      const vite = await createViteServer({
        server: { middlewareMode: true },
        appType: 'spa',
      });
      app.use(vite.middlewares);
    } catch (viteErr) {
      if (fs.existsSync(distIndexPath)) {
        console.warn('Vite dev middleware unavailable; falling back to static dist/:', viteErr);
        useViteDevMiddleware = false;
      } else {
        throw viteErr;
      }
    }
  }

  if (!useViteDevMiddleware) {
    app.use(
      express.static(distPath, {
        setHeaders: (res, filePath) => {
          if (
            filePath.endsWith('index.html') ||
            filePath.endsWith('sw.js') ||
            filePath.endsWith('sw-push.js') ||
            filePath.endsWith('manifest.webmanifest')
          ) {
            res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
            res.setHeader('Pragma', 'no-cache');
            res.setHeader('Expires', '0');
          }
        },
      })
    );
    app.get('*', (_req, res) => {
      res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
      res.setHeader('Pragma', 'no-cache');
      res.setHeader('Expires', '0');
      if (!fs.existsSync(distIndexPath)) {
        res.status(503).send('Application build not found.');
        return;
      }
      res.sendFile(distIndexPath);
    });
  }

  const httpServer = http.createServer(app);
  setupCommunityWebSocket({
    httpServer,
    sessionCookieName: SESSION_COOKIE_NAME,
  });

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
