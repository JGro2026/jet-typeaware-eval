"""DIB General Ledger Data (Gronewald et al. 2026) mapped onto the DA4JET Paper 1 protocol.

Unit of scoring: posting line (as the DATEV rows). Unit of evaluation: journal entry
(posting_id), entry score = max over its lines. Rule attributes (promptly, weekend, nwh,
top_n, high_cash) and label/marking are never used as features.
"""
import numpy as np, pandas as pd
from scipy.stats import rankdata

from dibpaths import DATA_CSV as CSV
RULE_COLS = ['promptly', 'weekend', 'nwh', 'top_n', 'high_cash']
CAT_COLS = ['gl_account', 'contra', 'cd_flag', 'tax_rate', 'user']   # currency -> cd_flag
EMB_DIMS = [10, 10, 1, 1, 1]


def load():
    df = pd.read_csv(CSV, sep=';', dtype={'ref_id': str, 'posting_id': str, 'text': str})
    df['row'] = np.arange(len(df))
    doc = pd.to_datetime(df.document_date); ent = pd.to_datetime(df.entered_date)
    post = pd.to_datetime(df.posting_date)
    df['lag_days'] = (ent - doc).dt.total_seconds() / 86400.0
    df['hour'] = ent.dt.hour + ent.dt.minute / 60 + ent.dt.second / 3600
    df['wd'] = ent.dt.weekday
    df['doc_outside_fy'] = (doc.dt.year != post.dt.year).astype(int)
    cents = (df.amount * 100).round().astype(np.int64)
    df['round_amt'] = (cents % 1000 == 0).astype(int)          # multiple of EUR 10
    df['text_present'] = df.text.fillna('').str.len().gt(0).astype(int)
    # contra account: largest-amount line on the opposite side of the same entry
    g = (df.sort_values(['posting_id', 'cd_flag', 'amount', 'gl_account'],
                        ascending=[True, True, False, True])
           .drop_duplicates(['posting_id', 'cd_flag'])[['posting_id', 'cd_flag', 'gl_account']])
    g['cd_flag'] = g.cd_flag.map({'C': 'D', 'D': 'C'})
    g = g.rename(columns={'gl_account': 'contra'})
    df = df.merge(g, on=['posting_id', 'cd_flag'], how='left').sort_values('row').reset_index(drop=True)
    df['contra'] = df.contra.fillna(-1).astype(int)
    df['eid'] = pd.factorize(df.posting_id)[0]
    return df


def entries(df):
    e = df.groupby('eid').agg(label=('label', 'max'), marking=('marking', 'max'),
                              top_n=('top_n', 'max'), high_cash=('high_cash', 'max'),
                              promptly=('promptly', 'max'), weekend=('weekend', 'max'),
                              nwh=('nwh', 'max'), n_lines=('row', 'size'))
    return e


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / s if s > 0 else np.zeros_like(x)


def numeric_block(df):
    h = 2 * np.pi * df.hour.values / 24; w = 2 * np.pi * df.wd.values / 7
    cols = {'log_amount': _z(np.log1p(df.amount.values)), 'lag_days': _z(df.lag_days.values),
            'hour_sin': _z(np.sin(h)), 'hour_cos': _z(np.cos(h)),
            'wd_sin': _z(np.sin(w)), 'wd_cos': _z(np.cos(w)),
            'round_amt': df.round_amt.values.astype(float),
            'doc_outside_fy': df.doc_outside_fy.values.astype(float),
            'text_present': df.text_present.values.astype(float)}
    return pd.DataFrame(cols)


def e_base(df):
    parts = [numeric_block(df)]
    for c in ['cd_flag', 'tax_rate', 'user', 'gl_account']:
        parts.append(pd.get_dummies(df[c].astype(str), prefix=c, dtype=float))
    cnt = df.contra.map(df.contra.value_counts())
    parts.append(pd.DataFrame({'contra_logcount': np.log1p(cnt.values)}))
    vc = df.contra.value_counts()
    top = sorted(vc.index, key=lambda v: (-vc[v], v))[:20]
    parts.append(pd.DataFrame({f'contra_{t}': (df.contra == t).astype(float).values for t in top}))
    parts.append(pd.DataFrame({'contra_other': (~df.contra.isin(top)).astype(float).values}))
    X = pd.concat(parts, axis=1)
    return X.values.astype(np.float64), list(X.columns)


def e_emb(df):
    codes, vocab = [], []
    for c in CAT_COLS:
        levels = np.sort(df[c].astype(str).unique())
        m = {v: i for i, v in enumerate(levels)}
        codes.append(df[c].astype(str).map(m).values); vocab.append(len(levels))
    return np.stack(codes, 1).astype(np.int64), numeric_block(df).values.astype(np.float32), vocab


def pct(x):
    return rankdata(x, method='average') / len(x)


def robust_z(x, grp, min_n=30):
    x = pd.Series(np.asarray(x, float)); grp = pd.Series(np.asarray(grp))
    gmed = x.median(); gmad = (x - gmed).abs().median(); gmad = gmad if gmad > 0 else 1.0
    med = x.groupby(grp).transform('median'); n = x.groupby(grp).transform('size')
    mad = (x - med).abs().groupby(grp).transform('median')
    small = n < min_n
    med = med.where(~small, gmed); mad = mad.where(~small, gmad)
    mad = mad.where(mad > 0, gmad)
    return ((x - med).abs() / mad).values


def decomp(df, extended=False):
    """Frozen DECOMP with the field mapping amount->s_amt, entry lag->s_pay.
    extended=True adds two post-hoc components outside the protocol:
    s_when = -log P(weekday x hour cell | user), s_lagu = robust z of lag within user."""
    N = len(df); a = df.gl_account.values; g = df.contra.values; u = df.user.values
    comp = {}
    comp['s_amt'] = robust_z(df.amount.values, a)
    comp['s_pay'] = robust_z(df.lag_days.values, a)
    rar = []
    for c in CAT_COLS:
        cnt = df[c].map(df[c].value_counts()).values
        rar.append(-np.log(cnt / N))
    comp['s_cat'] = np.max(np.stack(rar, 1), 1)
    ca = pd.Series(a).map(pd.Series(a).value_counts()).values
    cg = pd.Series(g).map(pd.Series(g).value_counts()).values
    pair = pd.Series(list(zip(a, g)))
    cag = pair.map(pair.value_counts()).values
    expected = ca * cg / N
    comp['s_pair'] = np.maximum(0, -np.log((cag + 0.5) / (expected + 0.5)))
    K = df.gl_account.nunique()
    cu = pd.Series(u).map(pd.Series(u).value_counts()).values
    ua = pd.Series(list(zip(u, a))); cua = ua.map(ua.value_counts()).values
    comp['s_user'] = -np.log((cua + 0.5) / (cu + 0.5 * K))
    if extended:
        cell = df.wd.values * 24 + np.floor(df.hour.values).astype(int)
        uc = pd.Series(list(zip(u, cell))); cuc = uc.map(uc.value_counts()).values
        comp['s_when'] = -np.log((cuc + 0.5) / (cu + 0.5 * 168))
        comp['s_lagu'] = robust_z(df.lag_days.values, u)
    P = {k: pct(v) for k, v in comp.items()}
    score = np.max(np.stack(list(P.values()), 1), 1)
    return score, P


def entry_scores(line_score, eid):
    s = pd.Series(line_score).groupby(eid).max()
    return s.values


def link_mismatch(df):
    """Post-hoc diagnostic outside the protocol: for entries that clear an earlier entry
    (ref_id = posting_id of the original), count mismatches with the original:
    different user, different item text, different gross amount. Returns an entry-level score 0..3."""
    orig = df.groupby('posting_id').agg(o_text=('text', 'first'), o_user=('user', 'first'), o_gross=('amount', 'max'))
    c = df[df.ref_id != '0'].groupby('eid').agg(ref=('ref_id', 'first'), text=('text', 'first'),
                                                 user=('user', 'first'), gross=('amount', 'max')).join(orig, on='ref')
    mis = ((c.text != c.o_text).astype(int) + (c.user != c.o_user).astype(int)
           + ((c.gross - c.o_gross).abs() > 0.01).astype(int))
    out = np.zeros(df.eid.nunique()); out[mis.index.values] = mis.values
    return out
