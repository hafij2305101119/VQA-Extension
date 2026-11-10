"""Findings sheet — HTML and PDF, identical in layout to the notebook's §5d.

Only fields the pipeline actually produces are printed. Everything a real
endoscopy report carries that this system has no evidence for — patient
identifiers, sedation, consent, endoscopist, scope serial, approver signature —
is omitted rather than filled with placeholders.
"""
from __future__ import annotations

import base64
import html as _html
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from .findings import Findings, fmt as _fmt, vqa_rows

PRODUCT_NAME = C.PRODUCT_NAME
PRODUCT_SUITE = C.PRODUCT_SUITE
PRODUCT_VERSION = C.PRODUCT_VERSION
REPORTED_BY = C.REPORTED_BY
MODEL_STACK = C.MODEL_STACK
DISCLAIMER = C.DISCLAIMER
RESTRICT_ANSWERS = C.RESTRICT_ANSWERS
VQA_GROUPS = C.QUESTION_GROUPS


def slug(s):
    return str(s).strip().replace(" ", "_").replace("/", "_")


def report_stamp(dt=None):
    return C.stamp(dt)


def vqa_frame(f):
    import pandas as pd
    return pd.DataFrame(vqa_rows(f))


# ── evidence extraction — every line traces back to a model output ──────────
def _site_of(f):
    if f.predicted_class=="Esophagitis": return "Esophagus"
    p=(f.procedure or "").lower()
    if "gastro" in p or "upper" in p or "esophag" in p: return "Upper GI tract"
    return "Colon" if "colono" in p else "Not reported"

def _procedure_label(f):
    p=(f.procedure or "").strip()
    return p.capitalize() if p and p.lower() not in ("none","nan") else "Endoscopy"

def _runner_up(f):
    if not f.class_probs: return None
    o=sorted(f.class_probs.items(),key=lambda kv:-kv[1])
    return o[1] if len(o)>1 else None

def _conf(f,key):
    """Confidence attached to one VQA answer, with provenance flags."""
    if key not in f.vqa: return "not asked"
    v=f.vqa[key]; s=f"{100*v['confidence']:.0f}% answer confidence"
    n=v.get("n_legal_answers")
    if n: s+=f", 1 of {n} legal answers"
    if v.get("in_coverage") is False: s+=" — question never paired with this class in training"
    return s

def report_items(case):
    """Ordered findings, each {text, evidence:[(label,value)]}. Purely derived
    from stage outputs — nothing is asserted that a model did not emit."""
    f=case["findings"]; m=case.get("seg_metrics"); out=[]

    ru=_runner_up(f)
    ev=[("Stage 1 classifier", f"Swin-T · top-1 {f.predicted_class} at {100*f.class_confidence:.1f}%")]
    if ru: ev.append(("Runner-up class", f"{ru[0]} at {100*ru[1]:.1f}%"))
    ev.append(("Attribution", "Grad-CAM localisation map attached"))
    if f.n_findings is not None: ev.append(("Findings counted", f"{f.n_findings} ({_conf(f,'n_findings')})"))
    ev.append(("Finding status", f"{_fmt(f.has_finding)} — basis: {f.finding_basis}"))
    if f.easy_to_detect is not None:
        ev.append(("Detection difficulty", f"{'easy' if f.easy_to_detect else 'not easy'} to detect "
                                           f"({_conf(f,'easy_detect')})"))
    out.append({"text": (f"Automated classification: {f.predicted_class}."
                         if f.predicted_class!="Normal" else
                         "Automated classification: no abnormality class assigned (Normal)."),
                "evidence":ev})

    if f.has_finding and f.abnormalities:
        out.append({"text": f"Abnormality reported: {', '.join(f.abnormalities)}.",
                    "evidence":[("Stage 2 VQA","Swin-T + DistilBERT cross-modal head"),
                                ("Answer confidence", _conf(f,"abnormalities")),
                                ("Image region", _fmt(f.location,"not reported")),
                                ("Colour", _fmt(f.colors,"not reported"))]})

    poly=[]
    if f.n_polyps_vqa is not None: poly.append(("Polyps counted (VQA)", f"{f.n_polyps_vqa} ({_conf(f,'n_polyps')})"))
    if f.polyp_type:              poly.append(("Morphology", f"{f.polyp_type} ({_conf(f,'polyp_type')})"))
    if f.polyp_size_vqa:          poly.append(("Reported size", f"{f.polyp_size_vqa} ({_conf(f,'polyp_size')})"))
    if f.polyps_removed is not None:
        poly.append(("All polyps removed", f"{f.polyps_removed_raw} ({_conf(f,'polyps_removed')})"))
    elif f.polyps_removed_raw:
        poly.append(("All polyps removed", f"{f.polyps_removed_raw} ({_conf(f,'polyps_removed')})"))
    if poly:
        n=f.n_polyps_vqa
        txt=("No polyp reported in this frame." if n==0 else
             f"{n} polyp{'s' if (n or 0)!=1 else ''} reported in this frame." if n is not None else
             "Polyp characteristics reported for this frame.")
        out.append({"text":txt,"evidence":poly})

    if f.n_polyps_detected:
        out.append({"text": f"{f.n_polyps_detected} lesion candidate"
                            f"{'s' if f.n_polyps_detected!=1 else ''} localised by the detector.",
                    "evidence":[("Stage 3 detector","YOLOv8s, confidence threshold 0.25"),
                                ("Highest box confidence", f"{f.detection_confidence:.2f}"),
                                ("Box quadrant", _fmt(f.detection_quadrant,"not reported")),
                                ("Cross-check", f"VQA polyp count {_fmt(f.n_polyps_vqa)}")]})
    elif f.predicted_class=="Polyps" or case.get("kind")=="kvasir-seg":
        out.append({"text":"No lesion candidate exceeded the detector confidence threshold.",
                    "evidence":[("Stage 3 detector","YOLOv8s, confidence threshold 0.25"),
                                ("Cross-check", f"VQA polyp count {_fmt(f.n_polyps_vqa)}")]})

    if f.area_pct_mucosa is not None:
        out.append({"text": f"Segmented lesion burden {f.area_pct_mucosa:.2f}% of visible mucosa "
                            f"({f.size_category}).",
                    "evidence":[("Stage 4 segmenter", f"U-Net++ / EfficientNet-B2 · {f.seg_model} variant"),
                                ("Size bucket","pixel-area proxy, not a calibrated measurement"),
                                ("VQA size answer", _fmt(f.polyp_size_vqa))]})

    if m:
        out.append({"text": "Agreement with the expert reference mask for this frame: "
                            f"Dice {m['dice']:.3f}, IoU {m['iou']:.3f}.",
                    "evidence":[("Pixel precision / recall", f"{m['pixel_precision']:.3f} / {m['pixel_recall']:.3f}"),
                                ("HD95","not defined (empty mask)" if np.isnan(m["hd95_px"]) else f"{m['hd95_px']:.1f} px"),
                                ("Expert burden", f"{m['gt_area_pct_mucosa']:.2f}% of mucosa"),
                                ("Predicted burden", f"{m['pred_area_pct_mucosa']:.2f}% "
                                                     f"(error {m['area_abs_error_pct']:.2f} pp)")]})

    if f.instruments:
        out.append({"text": f"Instrument visible in field: {', '.join(f.instruments)}.",
                    "evidence":[("Stage 2 VQA", _conf(f,"instruments")),
                                ("Instruments counted", _fmt(f.n_instruments)),
                                ("Image region", _fmt(f.instrument_location,"not reported"))]})
    if f.landmarks:
        out.append({"text": f"Anatomical landmark reported: {', '.join(f.landmarks)}.",
                    "evidence":[("Stage 2 VQA", _conf(f,"landmarks")),
                                ("Image region", _fmt(f.landmark_location,"not reported")),
                                ("Colour", _fmt(f.landmark_colors,"not reported"))]})

    q=[]
    if f.box_artefact is not None: q.append(("Green/black box artefact", f"{'present' if f.box_artefact else 'absent'} ({_conf(f,'artefact_box')})"))
    if f.text_present is not None: q.append(("Text overlay", f"{'present' if f.text_present else 'absent'} ({_conf(f,'text')})"))
    if q:
        flag=(f.box_artefact is True) or (f.text_present is True)
        out.append({"text":("Frame carries an overlay artefact that may bias the attribution map."
                            if flag else "No overlay artefact reported for this frame."),
                    "evidence":q})

    if f.has_finding is False and not f.n_polyps_detected and (f.area_pct_mucosa or 0)<0.1:
        out.append({"text":"No abnormality, lesion candidate or segmented burden reported for this frame.",
                    "evidence":[("Specificity check","all three lesion heads returned negative"),
                                ("Finding status", f"{_fmt(f.has_finding)} — basis: {f.finding_basis}")]})

    if f.out_of_coverage:
        out.append({"text":"Some questions were answered outside the class coverage of the "
                           "training metadata; treat those answers as unvalidated.",
                    "evidence":[("Questions affected", ", ".join(f.out_of_coverage)),
                                ("Reason","the dataset never pairs these questions with this class"),
                                ("Answer-space restriction",
                                 "on — each answer drawn from that question's legal set" if RESTRICT_ANSWERS
                                 else "off — answers drawn from the full vocabulary"),
                                ("Effect","the answer is still in-vocabulary but has no training support here")]})
    return out

# ── figure embedding ────────────────────────────────────────────────────────
FIG_CAPTIONS={"input":"Acquired frame","gradcam":"Stage 1 — Grad-CAM attribution",
              "detection":"Stage 3 — detector output","segmentation":"Stage 4 — segmentation overlay"}

def _b64(p, max_w=760):
    p=Path(p)
    if not p.exists(): return None
    img=cv2.imread(str(p))
    if img is None: return None
    if img.shape[1]>max_w:
        s=max_w/img.shape[1]; img=cv2.resize(img,(max_w,int(img.shape[0]*s)),interpolation=cv2.INTER_AREA)
    ok,buf=cv2.imencode(".png",img)
    return base64.b64encode(buf).decode() if ok else None

# ── HTML ────────────────────────────────────────────────────────────────────
_CSS = """
@page { size: A4; margin: 12mm 10mm; }
:root{ --accent:#C1592B; --rule:#7a7a7a; --ink:#111; --label-w:150px; }
*{ box-sizing:border-box; }
html,body{ margin:0; padding:0; }
body{ font-family:"Helvetica Neue", Arial, "Liberation Sans", sans-serif; color:var(--ink);
      font-size:10.5pt; line-height:1.34; -webkit-print-color-adjust:exact; print-color-adjust:exact; }
.sheet{ width:190mm; margin:0 auto; }
.masthead{ display:flex; align-items:center; justify-content:space-between; gap:10px;
           border-bottom:2.5px solid var(--accent); padding-bottom:5px; }
.brand{ display:flex; flex-direction:column; }
.brand .mark{ background:var(--accent); color:#fff; font-weight:700; font-size:13pt;
              letter-spacing:0.3px; padding:3px 9px; border-radius:2px; white-space:nowrap; }
.brand .sub{ font-size:7.6pt; color:#666; padding-left:2px; padding-top:2px; letter-spacing:0.2px; }
.masthead h1{ font-size:18pt; margin:0; letter-spacing:-0.2px; text-align:center; flex:1 1 auto; }
.masthead .procedure-type{ font-size:14.5pt; font-weight:700; color:var(--accent); white-space:nowrap; }
.meta{ display:grid; grid-template-columns:var(--label-w) 1fr; row-gap:3px; column-gap:6px;
       padding:7px 0; border-bottom:1.5px solid var(--rule); }
.meta b{ font-weight:700; }
h2.section{ color:var(--accent); font-size:14.5pt; margin:11px 0 4px; }
.site{ font-weight:700; margin-top:4px; }
ol.findings{ margin:0 0 0 15px; padding:0; }
ol.findings > li{ margin:4px 0; page-break-inside:avoid; break-inside:avoid; }
.subhead{ font-weight:700; margin-top:2px; }
.evgrid{ display:grid; grid-template-columns:1fr 1fr; gap:1px 16px; margin-left:3px; }
.evgrid .intv{ font-size:9.5pt; }
.evgrid .k{ font-weight:700; }
table.tech, table.vqa{ width:100%; border-collapse:collapse; font-size:9.3pt; margin-top:4px; }
table.tech th, table.tech td, table.vqa th, table.vqa td{ border:1px solid #ccc; padding:3px 5px; text-align:left; }
table.tech th, table.vqa th{ background:#f3f3f3; font-weight:700; }
table.vqa tr.grp td{ background:#faeee9; font-weight:700; color:var(--accent); }
.strip{ display:grid; grid-template-columns:repeat(2,1fr); gap:7px; margin-top:8px; }
figure{ margin:0; page-break-inside:avoid; break-inside:avoid; }
figure img{ width:100%; height:auto; display:block; border:1px solid #ddd; background:#eee; }
figcaption{ text-align:center; font-size:9.5pt; padding-top:2px; }
.signoff{ margin-top:10px; border-top:1.5px solid var(--rule); padding-top:5px;
          display:grid; grid-template-columns:1fr 1fr; gap:2px 20px; page-break-inside:avoid; }
.kv{ display:flex; padding:2px 0; }
.kv .k{ flex:0 0 104px; font-weight:700; }
.kv .v::before{ content:": "; }
.disclaimer{ font-size:8.5pt; color:#555; margin-top:7px; }
.footnote{ font-size:8.2pt; color:#666; margin:3px 0 0; }
"""

def _ts(case):
    """Generation time for this run — the sheet is stamped with the moment the
    report was produced, not the moment the module was imported."""
    return report_stamp(case.get("generated_at"))


def _meta_rows(case):
    f=case["findings"]
    return [("Procedure Date", _ts(case)),
            ("Report ID", f"{PRODUCT_NAME}-{slug(case['case_id']).upper()}"),
            ("Study Frame", Path(case["image_path"]).name),
            ("Source", case.get("source_label") or "Uploaded frame"),
            ("Procedure Type", _procedure_label(f)),
            ("Analysis Suite", f"{PRODUCT_SUITE} {PRODUCT_VERSION}"),
            ("Model Stack", MODEL_STACK)]

def build_report_html(case, embed_images=True):
    f=case["findings"]; e=_html.escape
    rows="\n".join(f"<b>{e(k)}</b><span>: {e(str(v))}</span>" for k,v in _meta_rows(case))

    lis=[]
    for it in report_items(case):
        ev="".join(f'<div class="intv"><span class="k">{e(k)}:</span> {e(str(v))}</div>'
                   for k,v in it["evidence"])
        lis.append(f'<li>{e(it["text"])}<div class="subhead">Supporting evidence :</div>'
                   f'<div class="evgrid">{ev}</div></li>')
    findings_html="\n".join(lis)

    vq=""
    if f.vqa:
        df=vqa_frame(f)
        for grp in VQA_GROUPS:
            sub=df[df["group"]==grp]
            if not len(sub): continue
            vq+=f'<tr class="grp"><td colspan="3">{e(grp)}</td></tr>'
            for _,r in sub.iterrows():
                flag=" ⚠" if r.get("trained_on_this_class") is False else ""
                star="*" if r.get("restriction_changed_answer") else ""
                vq+=(f'<tr><td>{e(r["question"])}{flag}</td>'
                     f'<td><b>{e(str(r["answer"]))}</b>{star}</td>'
                     f'<td>{100*r["confidence"]:.0f}%</td>'
                     f'<td>{e(str(r["legal_answers"]))}</td></tr>')
        vq=(f'<h2 class="section">Visual question answering — full response set</h2>'
            f'<table class="vqa"><tr><th style="width:57%">Question</th>'
            f'<th style="width:24%">Answer</th><th style="width:9%">Conf.</th>'
            f'<th style="width:10%">Answer space</th></tr>{vq}</table>'
            f'<p class="footnote">Each answer is selected from the legal answer set the dataset '
            f'defines for that question; * marks an answer that unrestricted decoding would have '
            f'got wrong, ⚠ marks a question the training metadata never pairs with this class.</p>')

    top=sorted(f.class_probs.items(),key=lambda kv:-kv[1]) if f.class_probs else []
    tech="".join(f"<tr><td>{e(k)}</td><td>{100*v:.2f}%</td></tr>" for k,v in top)

    figs=""
    for key,cap in FIG_CAPTIONS.items():
        p=case["figures"].get(key)
        if p is None or not Path(p).exists(): continue
        if embed_images:
            b=_b64(p)
            if not b: continue
            src=f"data:image/png;base64,{b}"
        else: src=Path(p).as_uri()
        figs+=f'<figure><img src="{src}" alt="{e(cap)}"><figcaption>{e(cap)}</figcaption></figure>'

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Endoscopy Findings Sheet — {e(case['title'])}</title>
<style>{_CSS}</style></head>
<body><div class="sheet">

  <div class="masthead">
    <div class="brand">
      <span class="mark">{e(PRODUCT_NAME)}™</span>
      <span class="sub">Endoscopy Analysis Suite {e(PRODUCT_VERSION)}</span>
    </div>
    <h1>Endoscopy Findings Sheet</h1>
    <div class="procedure-type">{e(_procedure_label(f))}</div>
  </div>

  <div class="meta">{rows}</div>

  <h2 class="section">Finding</h2>
  <div class="site">{e(_site_of(f))}:</div>
  <ol class="findings">
{findings_html}
  </ol>

  {vq}

  <h2 class="section">Classifier output</h2>
  <table class="tech"><tr><th>Class</th><th>Probability</th></tr>{tech}</table>

  <h2 class="section">Captured evidence</h2>
  <div class="strip">{figs}</div>

  <div class="signoff">
    <div class="kv"><span class="k">Reported by</span><span class="v">{e(REPORTED_BY)}</span></div>
    <div class="kv"><span class="k">Printed on</span><span class="v">{e(_ts(case))}</span></div>
  </div>

  <p class="disclaimer">{e(DISCLAIMER)}</p>
</div></body></html>"""

# ── PDF ─────────────────────────────────────────────────────────────────────
def _pdf_weasyprint(html_str, out_pdf):
    from weasyprint import HTML
    HTML(string=html_str).write_pdf(str(out_pdf)); return "weasyprint"

def _pdf_reportlab(case, out_pdf):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                    Image as RLImage, KeepTogether)
    f=case["findings"]; ACC=colors.HexColor("#C1592B"); ss=getSampleStyleSheet()
    body =ParagraphStyle("body",parent=ss["BodyText"],fontName="Helvetica",fontSize=9.3,leading=12)
    ev   =ParagraphStyle("ev",parent=body,fontSize=8.8,leading=11)
    small=ParagraphStyle("small",parent=body,fontSize=8.5,leading=10.5,textColor=colors.HexColor("#555555"))
    h2   =ParagraphStyle("h2",parent=ss["Heading2"],fontName="Helvetica-Bold",fontSize=12.5,
                         textColor=ACC,spaceBefore=9,spaceAfter=3)
    doc=SimpleDocTemplate(str(out_pdf),pagesize=A4,leftMargin=12*mm,rightMargin=12*mm,
                          topMargin=12*mm,bottomMargin=12*mm,
                          title=f"Endoscopy Findings Sheet — {case['title']}",author=REPORTED_BY)
    S=[]
    # branded band: wordmark block · centred title · procedure type
    mark=Table([[Paragraph(f'<font color="white"><b>{PRODUCT_NAME}™</b></font>',
                           ParagraphStyle("mk",parent=body,fontSize=12,leading=15,alignment=1))]],
               colWidths=[38*mm],rowHeights=[7.4*mm])
    mark.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),ACC),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                              ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3)]))
    brand=Table([[mark],[Paragraph(f"Endoscopy Analysis Suite {PRODUCT_VERSION}",
                                   ParagraphStyle("sb",parent=small,fontSize=6.6,leading=8))]],
                colWidths=[38*mm])
    brand.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),("TOPPADDING",(0,1),(-1,1),1),
                               ("BOTTOMPADDING",(0,0),(-1,-1),0)]))
    head=Table([[brand,
                 Paragraph("Endoscopy Findings Sheet",
                           ParagraphStyle("t",parent=body,fontName="Helvetica-Bold",fontSize=16,
                                          leading=19,alignment=1)),
                 Paragraph(f'<font color="#C1592B"><b>{_procedure_label(f)}</b></font>',
                           ParagraphStyle("pt",parent=body,fontSize=13,leading=16,alignment=2))]],
               colWidths=[42*mm,102*mm,42*mm])
    head.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                              ("LINEBELOW",(0,0),(-1,-1),1.6,ACC),
                              ("BOTTOMPADDING",(0,0),(-1,-1),4)]))
    S+=[head,Spacer(1,6)]

    mt=Table([[Paragraph(f"<b>{k}</b>",body),Paragraph(f": {v}",body)] for k,v in _meta_rows(case)],
             colWidths=[40*mm,146*mm])
    mt.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),1),
                            ("BOTTOMPADDING",(0,0),(-1,-1),1),
                            ("LINEBELOW",(0,-1),(-1,-1),0.9,colors.HexColor("#7a7a7a"))]))
    S+=[mt,Spacer(1,3),Paragraph("Finding",h2),
        Paragraph(f"<b>{_site_of(f)}:</b>",body),Spacer(1,2)]

    for i,it in enumerate(report_items(case),1):
        cells=[Paragraph(f"<b>{_html.escape(k)}:</b> {_html.escape(str(v))}",ev) for k,v in it["evidence"]]
        if len(cells)%2: cells.append(Paragraph("",ev))
        grid=[[cells[j],cells[j+1]] for j in range(0,len(cells),2)]   # two-column evidence
        et=Table(grid,colWidths=[91*mm,91*mm])
        et.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),8),
                                ("TOPPADDING",(0,0),(-1,-1),0.5),("BOTTOMPADDING",(0,0),(-1,-1),0.5)]))
        S+=[KeepTogether([Paragraph(f"{i}. {_html.escape(it['text'])}",body),
                          Paragraph("<b>Supporting evidence :</b>",
                                    ParagraphStyle("sh",parent=body,leftIndent=8,spaceBefore=1)),
                          et]),Spacer(1,4)]

    if f.vqa:
        S+=[Paragraph("Visual question answering — full response set",h2)]
        rows=[[Paragraph("<b>Question</b>",ev),Paragraph("<b>Answer</b>",ev),
               Paragraph("<b>Conf.</b>",ev),Paragraph("<b>Answer space</b>",ev)]]
        style=[("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#cccccc")),
               ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f3f3f3")),
               ("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),2),
               ("BOTTOMPADDING",(0,0),(-1,-1),2)]
        df=vqa_frame(f)
        for grp in VQA_GROUPS:
            sub=df[df["group"]==grp]
            if not len(sub): continue
            r=len(rows); rows.append([Paragraph(f'<font color="#C1592B"><b>{grp}</b></font>',ev),"","",""])
            style+=[("SPAN",(0,r),(-1,r)),("BACKGROUND",(0,r),(-1,r),colors.HexColor("#FAEEE9"))]
            for _,q in sub.iterrows():
                flag=" *" if q.get("restriction_changed_answer") else ""
                warn=' <font color="#C44E52"><b>!</b></font>' if q.get("trained_on_this_class") is False else ""
                rows.append([Paragraph(_html.escape(q["question"])+warn,ev),
                             Paragraph(f"<b>{_html.escape(str(q['answer']))}</b>{flag}",ev),
                             Paragraph(f"{100*q['confidence']:.0f}%",ev),
                             Paragraph(str(q["legal_answers"]),ev)])
        t=Table(rows,colWidths=[102*mm,46*mm,17*mm,21*mm],repeatRows=1)
        t.setStyle(TableStyle(style)); S+=[t]
        S+=[Paragraph("Each answer is selected from the legal answer set the dataset defines for that "
                      "question; * marks an answer that unrestricted decoding would have got wrong, "
                      "! marks a question the training metadata never pairs with this class.",small)]

    if f.class_probs:
        S+=[Paragraph("Classifier output",h2)]
        rows=[["Class","Probability"]]+[[k,f"{100*v:.2f}%"] for k,v in
              sorted(f.class_probs.items(),key=lambda kv:-kv[1])]
        t=Table(rows,colWidths=[118*mm,68*mm])
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#cccccc")),
                               ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f3f3f3")),
                               ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                               ("FONTSIZE",(0,0),(-1,-1),9),("TOPPADDING",(0,0),(-1,-1),2),
                               ("BOTTOMPADDING",(0,0),(-1,-1),2)]))
        S+=[t]

    cells=[]
    for key,cap in FIG_CAPTIONS.items():
        p=case["figures"].get(key)
        if p is None or not Path(p).exists(): continue
        im=RLImage(str(p)); r=im.imageHeight/max(im.imageWidth,1)
        im.drawWidth=88*mm; im.drawHeight=88*mm*r
        cells.append([im,Paragraph(cap,ParagraphStyle("cap",parent=small,alignment=1))])
    if cells:
        S+=[Paragraph("Captured evidence",h2)]
        grid=[]
        for i in range(0,len(cells),2):
            pair=cells[i:i+2]
            grid.append([pair[0][0], pair[1][0] if len(pair)>1 else ""])
            grid.append([pair[0][1], pair[1][1] if len(pair)>1 else ""])
        t=Table(grid,colWidths=[92*mm,92*mm])
        t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("ALIGN",(0,0),(-1,-1),"CENTER"),
                               ("BOTTOMPADDING",(0,0),(-1,-1),3)]))
        S+=[t]

    S+=[Spacer(1,7),
        Table([[Paragraph(f"<b>Reported by</b>: {REPORTED_BY}",body),
                Paragraph(f"<b>Printed on</b>: {_ts(case)}",body)]],colWidths=[118*mm,68*mm],
              style=TableStyle([("LINEABOVE",(0,0),(-1,0),0.9,colors.HexColor("#7a7a7a")),
                                ("TOPPADDING",(0,0),(-1,-1),4)])),
        Spacer(1,4),Paragraph(DISCLAIMER,small)]
    doc.build(S); return "reportlab"

def write_report(case, outdir: Path) -> dict:
    """Write the findings sheet next to the run's figures.

    WeasyPrint is used when importable — the PDF is then a pixel-for-pixel render
    of the same HTML. ReportLab is the always-available fallback and produces the
    same sheet from the same data. The filename carries the generation timestamp
    so downloads never collide and are self-describing.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = (case.get("generated_at") or C.now()).strftime("%Y%m%d-%H%M%S")
    stem = f"{C.PRODUCT_NAME}_{slug(case['case_id'])}_{ts}"

    html_str = build_report_html(case)
    html_p = outdir / f"{stem}.html"
    html_p.write_text(html_str, encoding="utf-8")

    pdf_p = outdir / f"{stem}.pdf"
    engine, last = None, None
    for fn in (lambda: _pdf_weasyprint(html_str, pdf_p),
               lambda: _pdf_reportlab(case, pdf_p)):
        try:
            engine = fn()
            break
        except Exception as exc:      # engine missing or a system lib absent
            last = exc
    if engine is None:
        pdf_p = None

    return {"html": html_p, "pdf": pdf_p, "engine": engine,
            "download_stem": stem,
            "error": None if engine else f"{type(last).__name__}: {last}"}
