# Security Specification & Red Team Audit Plan

## 1. Data Invariants
1. **Email-Keyed User Profiles (`/users/{userEmail}`)**:
   - Every `/users/{userEmail}` document uses the lowercased, trimmed user email as its Document ID (`userEmail == userEmail.lower()`).
   - Writes (`create`, `update`, `delete`) are strictly restricted to verified Firebase Auth owners whose token email matches `userEmail` (`request.auth != null && request.auth.token.email_verified == true && request.auth.token.email.lower() == userEmail`).
   - Documents store profile metadata, XP/level/progress, `lastSyncAt`, and lightweight custom words (no audio recordings and no built-in dictionary duplication).
   - Reads (`get`, `list`) validate document structure (`existing().email is string && existing().username is string && existing().xp is number`) so subscriber counts, leaderboard rankings, and account restoration operate consistently on startup and refresh.
2. **Public PII-Free Leaderboard (`/leaderboard/{userId}`)**:
   - `/leaderboard/{userId}` MUST NEVER contain `email`, phone, or private files.
   - Writes to `/leaderboard/{userId}` must strictly enforce `existsAfter(/databases/$(database)/documents/users/$(request.auth.token.email.lower()))` so leaderboard entries are atomically synchronized with the email-keyed user document.
   - `xp`, `wordsCount`, `masteredCount`, `totalReviews`, `streak`, `levelNumber` are strictly bounded (`masteredCount <= wordsCount`, `xp >= 0`).
3. **Immutability & Temporal Integrity**:
   - `uid`, `email`, and `createdAt` are immutable after document creation.
   - `createdAt == request.time` on `create`; `updatedAt == request.time` and `lastSyncAt == request.time` on `create` and `update`.

## 2. The "Dirty Dozen" Adversarial Payloads
1. **Unauthenticated Write**: `setDoc('/users/a@b.com', {...})` with `auth = null` -> DENIED.
2. **Unverified Email Write**: `setDoc('/users/a@b.com', {...})` with `email_verified = false` -> DENIED.
3. **Cross-Tenant Write / Email Spoofing**: `setDoc('/users/victim@b.com', {...})` by `attacker@b.com` -> DENIED (`request.auth.token.email.lower() == userEmail`).
4. **Non-Lowercased / Poisoned Email ID**: `setDoc('/users/Victim@B.com', {...})` or path injection -> DENIED (`isValidEmailDocId(userEmail)`).
5. **Shadow / Ghost Field Injection**: `setDoc('/users/a@b.com', { ..., isAdmin: true })` -> DENIED (`keys().hasOnly(...)`).
6. **Shadow Field Update**: `updateDoc('/users/a@b.com', { role: 'admin' })` -> DENIED (`affectedKeys().hasOnly(...)`).
7. **UID Spoofing**: `setDoc('/users/a@b.com', { ..., uid: 'other-uid' })` -> DENIED (`data.uid == request.auth.uid`).
8. **ID Poisoning**: `getDoc('/leaderboard/' + 'a'.repeat(500))` or invalid chars -> DENIED (`isValidId(userId)`).
9. **Array Type/Size Poisoning**: `updateDoc('/users/a@b.com', { words: ['not-a-map'], ... })` or `words.size() > 2500` -> DENIED.
10. **Mastered > Words Invariant Poisoning**: `setDoc('/users/a@b.com', { wordsCount: 2, masteredCount: 99, ... })` -> DENIED.
11. **Client Timestamp Forgery**: `setDoc('/users/a@b.com', { createdAt: Timestamp(2020), updatedAt: serverTimestamp(), lastSyncAt: serverTimestamp() })` -> DENIED (`createdAt == request.time`).
12. **Orphaned Leaderboard Write**: `setDoc('/leaderboard/u1', ...)` without `/users/{email}` existing in the same batch -> DENIED (`existsAfter(/users/...)`).
13. **Cross-Tenant Direct Chat Read/Write**: Reading or writing `/directChats/{uidA_uidB}` or its `messages` by a 3rd user `uidC` -> DENIED (`request.auth.uid in participantIds`).
14. **Canonical Chat ID Forgery & Message Spoofing**: Creating `/directChats/fakeId` where `chatId != participantA + '_' + participantB`, or creating a message where `senderId != request.auth.uid`, or injecting non-audio URI in `audioDataUrl` -> DENIED.
