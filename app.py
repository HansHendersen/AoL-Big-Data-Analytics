"""
Dashboard Keberhasilan Mahasiswa (OULAD): Streamlit + Plotly
Jalankan:  streamlit run app.py
Data     :  folder dashboard_data/ (dibuat oleh export_dashboard_data.py)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Kompatibel dengan Streamlit lama (use_container_width) maupun baru (width="stretch", >= 1.50)
_v = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

st.set_page_config(page_title="Dashboard Keberhasilan Mahasiswa OULAD", layout="wide")

DATA = Path(__file__).parent / "dashboard_data"
ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]
COLORS = {"Distinction": "#2e7d32", "Pass": "#1976d2", "Fail": "#f59e0b", "Withdrawn": "#d32f2f"}
FEATURE_ID = {
    "last_activity_day": "Hari terakhir aktif di platform",
    "n_assessments": "Jumlah tugas yang dikumpulkan",
    "active_days": "Jumlah hari aktif belajar",
    "click_homepage": "Klik halaman utama",
    "avg_score": "Rata-rata nilai tugas",
    "has_assessment": "Pernah mengumpulkan tugas",
    "avg_score_tma": "Rata-rata nilai tugas besar (TMA)",
    "max_score": "Nilai tugas tertinggi",
    "min_score": "Nilai tugas terendah",
    "click_resource": "Klik materi (resource)",
    "click_quiz": "Klik kuis",
    "total_click": "Total klik",
    "distinct_sites": "Jumlah materi berbeda yang dibuka",
    "avg_click_per_active_day": "Rata-rata klik per hari aktif",
    "code_module": "Mata kuliah (modul)",
}
BEHAVIOR_ID = {
    "active_days": "Jumlah hari aktif",
    "total_click": "Total klik",
    "n_assessments": "Jumlah tugas dikumpulkan",
    "avg_score": "Rata-rata nilai tugas",
    "last_activity_day": "Hari terakhir aktif",
}


@st.cache_data
def load(name):
    p = DATA / name
    return pd.read_csv(p) if p.exists() else None


def need(df, name):
    if df is None:
        st.warning(f"File `dashboard_data/{name}` belum ada. Jalankan kode ekspor di notebook dulu.")
        return False
    return True


def stacked_share(df, by, order=None, height=360):
    """Bar 100% bertumpuk: proporsi final_result untuk tiap nilai kolom `by`."""
    t = (df.groupby(by, observed=True)["final_result"].value_counts(normalize=True)
           .unstack().reindex(columns=ORDER).fillna(0))
    if order is not None:
        t = t.reindex(order).dropna(how="all")
    long = t.reset_index().melt(id_vars=by, var_name="Hasil", value_name="Proporsi")
    fig = px.bar(long, y=by, x="Proporsi", color="Hasil", orientation="h",
                 color_discrete_map=COLORS, category_orders={"Hasil": ORDER},
                 text=long["Proporsi"].map(lambda v: f"{v:.0%}" if v >= 0.07 else ""))
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None,
                      legend=dict(orientation="h", y=-0.15))
    return fig, t


# ------------------------------------------------------------------ header
st.title("Dashboard Keberhasilan Mahasiswa")
st.caption("Open University Learning Analytics Dataset (OULAD) dengan 32.593 data mahasiswa-modul. "
           "Distinction = lulus dengan pujian, Pass = lulus, Fail = tidak lulus, "
           "Withdrawn = mengundurkan diri.")

students = load("students.csv")
if students is None:
    st.error("`dashboard_data/students.csv` tidak ditemukan. "
             "Jalankan Bagian A di `export_dashboard_data.py` pada notebook modeling.")
    st.stop()

# ------------------------------------------------------------------ filter
with st.sidebar:
    st.header("Filter")
    st.caption("Berlaku untuk tab 1 dan 2.")
    mods = st.multiselect("Modul", sorted(students.code_module.unique()),
                          default=sorted(students.code_module.unique()))
    pres = st.multiselect("Periode (presentation)", sorted(students.code_presentation.unique()),
                          default=sorted(students.code_presentation.unique()))
    ages = st.multiselect("Kelompok usia", sorted(students.age_band.dropna().unique()),
                          default=sorted(students.age_band.dropna().unique()))
    gender = st.multiselect("Gender", sorted(students.gender.unique()),
                            default=sorted(students.gender.unique()))

df = students[students.code_module.isin(mods) & students.code_presentation.isin(pres)
              & students.age_band.isin(ages) & students.gender.isin(gender)].copy()

tab1, tab2, tab3, tab4 = st.tabs(["1 · Gambaran Umum", "2 · Perilaku Belajar",
                                  "3 · Prediksi (Machine Learning)",
                                  "4 · Kemiripan Mahasiswa (Graph)"])

# ================================================================== TAB 1
with tab1:
    if df.empty:
        st.warning("Tidak ada data untuk kombinasi filter ini.")
    else:
        share = df.final_result.value_counts(normalize=True).reindex(ORDER).fillna(0)
        lulus = share["Pass"] + share["Distinction"]
        c = st.columns(4)
        c[0].metric("Mahasiswa-modul", f"{len(df):,}")
        c[1].metric("Lulus (Pass + Distinction)", f"{lulus:.1%}")
        c[2].metric("Tidak lulus (Fail)", f"{share['Fail']:.1%}")
        c[3].metric("Mengundurkan diri (Withdrawn)", f"{share['Withdrawn']:.1%}")

        col_pie, col_bar = st.columns([1, 2])
        with col_pie:
            st.subheader("Proporsi mahasiswa")
            dim_map = {"Hasil akhir": "final_result", "Modul": "code_module",
                       "Kelompok usia": "age_band", "Gender": "gender"}
            dim = dim_map[st.selectbox("Kelompokkan berdasarkan", list(dim_map))]
            cnt = df.groupby(dim).size().rename("jumlah").reset_index()
            if dim == "final_result":
                cnt = cnt.set_index(dim).reindex(ORDER).fillna(0).reset_index()
            figp = px.pie(cnt, names=dim, values="jumlah", hole=0.4, color=dim,
                          color_discrete_map=COLORS if dim == "final_result" else None)
            figp.update_traces(sort=False, textinfo="percent")
            figp.update_layout(height=296, margin=dict(l=0, r=0, t=10, b=0),
                               legend=dict(orientation="h", y=-0.05))
            st.plotly_chart(figp, **STRETCH)

        with col_bar:
            st.subheader("Proporsi hasil akhir per modul")
            fig, t = stacked_share(df, "code_module", order=None, height=380)
            t["lulus"] = t["Pass"] + t["Distinction"]
            fig.update_yaxes(categoryorder="array", categoryarray=t.sort_values("lulus").index.tolist())
            st.plotly_chart(fig, **STRETCH)

        st.subheader("Tren hasil akhir per periode")
        per = (df.groupby("code_presentation")["final_result"].value_counts(normalize=True)
                 .unstack().reindex(columns=ORDER).fillna(0).sort_index())
        lg = per.reset_index().melt(id_vars="code_presentation", var_name="Hasil", value_name="Proporsi")
        figl = px.line(lg, x="code_presentation", y="Proporsi", color="Hasil", markers=True,
                       color_discrete_map=COLORS,
                       category_orders={"Hasil": ORDER, "code_presentation": list(per.index)})
        figl.update_yaxes(tickformat=".0%", rangemode="tozero")
        figl.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.25),
                           xaxis_title="Periode (B = mulai Februari, J = mulai Oktober)", yaxis_title=None)
        st.plotly_chart(figl, **STRETCH)
        wd_hi, wd_lo = per["Withdrawn"].idxmax(), per["Withdrawn"].idxmin()
        st.caption(f"Tingkat mundur tertinggi terjadi pada periode **{wd_hi}** ({per.loc[wd_hi, 'Withdrawn']:.0%}) "
                   f"dan terendah pada **{wd_lo}** ({per.loc[wd_lo, 'Withdrawn']:.0%}).")

        worst, best = t["Withdrawn"].idxmax(), t["lulus"].idxmax()

# ================================================================== TAB 2
with tab2:
    if df.empty:
        st.warning("Tidak ada data untuk kombinasi filter ini.")
    else:
        left, right = st.columns(2)

        with left:
            st.subheader("Kapan mahasiswa terakhir aktif?")
            bucket_order = ["Tidak pernah aktif", "Hari ≤ 0", "Hari 1–30", "Hari 31–90",
                            "Hari 91–180", "Hari > 180"]
            d = df.copy()
            d["Terakhir aktif"] = pd.cut(d.last_activity_day,
                                         [-np.inf, 0, 30, 90, 180, np.inf],
                                         labels=bucket_order[1:]).astype(object)
            d["Terakhir aktif"] = d["Terakhir aktif"].fillna("Tidak pernah aktif")
            fig, tb = stacked_share(d, "Terakhir aktif", order=bucket_order)
            fig.update_layout(height=620)
            st.plotly_chart(fig, **STRETCH)

        with right:
            st.subheader("Pernah mengumpulkan tugas?")
            d["Tugas"] = np.where(d.n_assessments > 0, "Mengumpulkan ≥ 1 tugas",
                                  "Tidak pernah mengumpulkan tugas")
            fig2, tt = stacked_share(d, "Tugas", height=220)
            st.plotly_chart(fig2, **STRETCH)

            st.subheader("Rata-rata perilaku menurut hasil akhir")
            var = st.selectbox("Ukuran", list(BEHAVIOR_ID), format_func=BEHAVIOR_ID.get)
            m = df.groupby("final_result")[var].mean().reindex(ORDER).reset_index()
            fig3 = px.bar(m, x="final_result", y=var, color="final_result",
                          color_discrete_map=COLORS, text=m[var].round(1))
            fig3.update_layout(height=260, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title=None, yaxis_title=BEHAVIOR_ID[var])
            st.plotly_chart(fig3, **STRETCH)

        st.subheader(f"Tren {BEHAVIOR_ID[var].lower()} per periode")
        tr = df.groupby(["code_presentation", "final_result"])[var].mean().reset_index()
        figtr = px.line(tr, x="code_presentation", y=var, color="final_result", markers=True,
                        color_discrete_map=COLORS,
                        category_orders={"final_result": ORDER,
                                         "code_presentation": sorted(tr.code_presentation.unique())})
        figtr.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), legend_title_text="Hasil",
                            legend=dict(orientation="h", y=-0.25), xaxis_title="Periode",
                            yaxis_title=BEHAVIOR_ID[var])
        st.plotly_chart(figtr, **STRETCH)

        no_task = tt.loc["Tidak pernah mengumpulkan tugas"] if "Tidak pernah mengumpulkan tugas" in tt.index else None
        has_task = tt.loc["Mengumpulkan ≥ 1 tugas"] if "Mengumpulkan ≥ 1 tugas" in tt.index else None
        if no_task is not None and has_task is not None:
            early = tb.loc["Hari 1–30", "Withdrawn"] if "Hari 1–30" in tb.index else float("nan")

# ================================================================== TAB 3
with tab3:
    mm, fi = load("model_metrics.csv"), load("feature_importance.csv")
    pc, pr = load("per_class_f1.csv"), load("predictions.csv")

    if need(mm, "model_metrics.csv"):
        trained = mm[~mm.model.str.startswith("Baseline")].sort_values("macro_f1", ascending=False)
        base = mm[mm.model == "Baseline: kelas mayoritas"]
        top = trained.iloc[0]
        c = st.columns(4)
        c[0].metric("Model terbaik", top.model.split(" - ")[0])
        c[1].metric("Akurasi", f"{top.accuracy:.1%}",
                    delta=(f"{(top.accuracy - base.accuracy.iloc[0]) * 100:+.0f} poin vs tebakan "
                           f"kelas terbanyak" if len(base) else None))
        c[2].metric("Macro F1", f"{top.macro_f1:.2f}")
        c[3].metric("Fitur dipakai", "15 dari 49 kolom")

        st.subheader("Perbandingan model")
        long = mm.melt(id_vars="model", value_vars=["accuracy", "macro_f1"],
                       var_name="Metrik", value_name="Skor")
        fig = px.bar(long, y="model", x="Skor", color="Metrik", barmode="group", orientation="h",
                     category_orders={"model": mm.sort_values("macro_f1").model.tolist()},
                     color_discrete_sequence=["#1976d2", "#f59e0b"])
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None,
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, **STRETCH)

    if need(fi, "feature_importance.csv"):
        st.subheader("Apa yang paling menentukan hasil mahasiswa?")
        f = fi.sort_values("importance", ascending=False).head(12).copy()
        f["Fitur"] = f.fitur.map(lambda x: FEATURE_ID.get(x, x))
        fig = px.bar(f.iloc[::-1], x="importance", y="Fitur", orientation="h",
                     color_discrete_sequence=["#1976d2"],
                     text=f.iloc[::-1].importance.map(lambda v: f"{v:.1%}"))
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="Tingkat pengaruh", yaxis_title=None)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, **STRETCH)
        t3 = f.head(3)

    if need(pc, "per_class_f1.csv"):
        st.subheader("Seberapa baik model mengenali tiap kelompok?")
        pick = st.selectbox("Model", pc.model.tolist(),
                            index=pc.model.tolist().index("XGBoost - xgboost")
                            if "XGBoost - xgboost" in pc.model.tolist() else 0)
        row = pc[pc.model == pick].melt(id_vars="model", var_name="Hasil", value_name="F1")
        fig = px.bar(row, x="Hasil", y="F1", color="Hasil", color_discrete_map=COLORS,
                     category_orders={"Hasil": ORDER}, text=row.F1.round(2))
        fig.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_range=[0, 1])
        st.plotly_chart(fig, **STRETCH)
        st.caption("Skor F1 (0–1): makin tinggi makin baik. Kelompok Fail dan Distinction lebih sulit "
                   "ditebak karena jumlahnya lebih sedikit dan polanya mirip dengan kelompok Pass.")

    if need(pr, "predictions.csv"):
        st.subheader("Daftar peringatan dini")
        pr["risk"] = pr.get("p_Fail", 0) + pr.get("p_Withdrawn", 0)
        pr["berisiko"] = pr.actual.isin(["Fail", "Withdrawn"])
        thr = st.slider("Tandai mahasiswa jika risiko gagal atau mundur ≥", 0.30, 0.95, 0.70, 0.05)
        flag = pr[pr.risk >= thr]
        prec = flag.berisiko.mean() if len(flag) else float("nan")
        rec = flag.berisiko.sum() / max(pr.berisiko.sum(), 1)
        c = st.columns(3)
        c[0].metric("Mahasiswa ditandai", f"{len(flag):,} dari {len(pr):,}")
        c[1].metric("Ketepatan tanda", f"{prec:.0%}",
                    help="Dari yang ditandai, berapa persen benar-benar gagal atau mundur.")
        c[2].metric("Tertangkap", f"{rec:.0%}",
                    help="Dari seluruh mahasiswa yang gagal atau mundur, berapa persen berhasil ditandai.")

        cm = pd.crosstab(pr.actual, pr.predicted, normalize="index").reindex(index=ORDER, columns=ORDER).fillna(0)
        figcm = px.imshow(cm, text_auto=".0%", color_continuous_scale="Blues", zmin=0, zmax=1,
                          labels=dict(x="Prediksi model", y="Hasil sebenarnya", color="Proporsi"))
        figcm.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        a, b = st.columns([1, 1])
        a.plotly_chart(figcm, **STRETCH)
        b.markdown("**Mahasiswa dengan risiko tertinggi**")
        b.dataframe(flag.sort_values("risk", ascending=False).head(15)
                    [["id_student", "code_module", "predicted", "risk"]]
                    .rename(columns={"predicted": "prediksi", "risk": "risiko"})
                    .style.format({"risiko": "{:.0%}"}),
                    **STRETCH, hide_index=True, height=340)

    if mm is not None:
        with st.expander("Spark vs non-Spark: waktu pelatihan"):
            tm = mm[~mm.model.str.startswith("Baseline")].copy()
            tm["Mesin"] = np.where(tm.model.str.contains("Spark"), "Spark", "Non-Spark")
            figt = px.bar(tm.sort_values("fit_s"), y="model", x="fit_s", color="Mesin", orientation="h",
                          color_discrete_map={"Spark": "#f59e0b", "Non-Spark": "#1976d2"},
                          text=tm.sort_values("fit_s").fit_s.round(1))
            figt.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title="Detik (median 3 kali)", yaxis_title=None)
            st.plotly_chart(figt, **STRETCH)
            st.caption("Pada data sekecil ini (±26 ribu baris, 21 fitur), pelatihan tanpa Spark lebih "
                       "cepat karena Spark memiliki biaya koordinasi. Spark baru unggul bila data tidak "
                       "lagi muat di memori satu mesin.")

# ================================================================== TAB 4
with tab4:
    nodes, edges = load("graph_nodes.csv"), load("graph_edges.csv")
    if need(nodes, "graph_nodes.csv") and need(edges, "graph_edges.csv"):
        st.write("Setiap titik adalah satu mahasiswa; ukuran titik = jumlah klik di platform. Garis "
                 "menghubungkan mahasiswa yang pola belajarnya mirip (materi yang dibuka dan volume klik). "
                 "Tiap mahasiswa dihubungkan ke maksimal 3 mahasiswa paling mirip.")

        lo = round(float(np.floor(edges.weight.min() * 20) / 20), 2)
        thr = st.slider("Tampilkan koneksi dengan kemiripan ≥", lo, max(0.95, lo + 0.05), lo, 0.05,
                        help="Semakin tinggi, semakin sedikit koneksi: hanya pasangan yang sangat mirip.")
        e = edges[edges.weight >= thr].copy()

        pos = nodes.set_index("id")[["x", "y"]]
        res = nodes.set_index("id").final_result
        deg = pd.concat([e.source, e.target]).value_counts()
        nodes["koneksi"] = nodes["id"].map(deg).fillna(0).astype(int)

        def line_xy(sub):
            xs = np.column_stack([sub.source.map(pos.x), sub.target.map(pos.x), np.full(len(sub), np.nan)])
            ys = np.column_stack([sub.source.map(pos.y), sub.target.map(pos.y), np.full(len(sub), np.nan)])
            return xs.ravel(), ys.ravel()

        fig = go.Figure()
        for name, mask, w in [("Kemiripan rendah (< 0,4)", e.weight < 0.4, 0.8),
                              ("Kemiripan sedang (0,4–0,7)", (e.weight >= 0.4) & (e.weight < 0.7), 2.0),
                              ("Kemiripan tinggi (≥ 0,7)", e.weight >= 0.7, 4.0)]:
            sub = e[mask]
            if len(sub):
                x, y = line_xy(sub)
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, hoverinfo="skip",
                                         line=dict(width=w, color="rgba(140,140,140,0.40)")))

        nodes["size"] = 8 + 22 * np.log1p(nodes.total_clicks) / np.log1p(nodes.total_clicks).max()
        for g in ORDER:
            sub = nodes[nodes.final_result == g]
            fig.add_trace(go.Scatter(
                x=sub.x, y=sub.y, mode="markers", name=g,
                marker=dict(size=sub["size"], color=COLORS[g], opacity=0.85,
                            line=dict(width=0.5, color="white")),
                text=[f"{r.final_result}<br>Total klik: {r.total_clicks:,.0f}<br>Koneksi: {r.koneksi}"
                      for r in sub.itertuples()],
                hoverinfo="text"))
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=-0.02))
        st.plotly_chart(fig, **STRETCH)

        if e.empty:
            st.warning("Tidak ada koneksi pada ambang kemiripan ini. Turunkan nilai slider.")
        else:
            r1, r2 = e.source.map(res), e.target.map(res)
            same_mask = (r1 == r2)
            same = same_mask.mean()
            p = nodes.final_result.value_counts(normalize=True)
            chance = float((p ** 2).sum())
            sim_same = e.weight[same_mask].mean()
            sim_diff = e.weight[~same_mask].mean()

            c = st.columns(4)
            c[0].metric("Mahasiswa / koneksi", f"{len(nodes)} / {len(e)}")
            c[1].metric("Rata-rata kemiripan", f"{e.weight.mean():.2f}")
            c[2].metric("Koneksi sesama hasil akhir", f"{same:.0%}",
                        help="Persentase garis yang menghubungkan dua mahasiswa dengan hasil akhir sama.")
            c[3].metric("Bila acak murni", f"{chance:.0%}")

            pairs = pd.concat([pd.DataFrame({"a": r1.to_numpy(), "b": r2.to_numpy()}),
                                pd.DataFrame({"a": r2.to_numpy(), "b": r1.to_numpy()})], ignore_index=True)
            mix = (pd.crosstab(pairs.a, pairs.b, normalize="index")
                   .reindex(index=ORDER, columns=ORDER).fillna(0))
            figm = px.imshow(mix, text_auto=".0%", color_continuous_scale="Blues", zmin=0, zmax=1,
                             labels=dict(x="Hasil akhir teman terdekatnya", y="Hasil akhir mahasiswa",
                                         color="Proporsi"))
            figm.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
            st.subheader("Mahasiswa mirip dengan siapa?")
            st.caption("Baca per baris: dari mahasiswa dengan hasil akhir di baris, berapa persen teman "
                       "terdekatnya (dalam jaringan) berhasil akhir seperti di kolom.")
            st.plotly_chart(figm, **STRETCH)

            diag = pd.Series({k: mix.loc[k, k] for k in ORDER})
            most, least = diag.idxmax(), diag.idxmin()

        st.caption("Catatan: kemiripan dihitung 70% dari pola materi yang dibuka dan 30% dari volume klik. "
                   "Graph dibangun dari sampel mahasiswa yang seimbang per kelas (bukan proporsi populasi "
                   "asli), sehingga proporsinya berbeda dari tab 1.")

st.divider()
st.caption("Catatan metodologi: fitur perilaku dihitung dari seluruh periode kursus (CUTOFF = None). "
           "Untuk peringatan dini yang benar-benar dipakai saat semester berjalan, ETL perlu dijalankan "
           "ulang dengan CUTOFF tertentu (misalnya hari ke-60).")
