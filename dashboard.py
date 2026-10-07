import json
import os
import pandas as pd
import streamlit as st

# استدعاء المحرك الرئيسي (تأكد من وجود ملف نظام_بدون_مفتاح_V0_1.py في نفس المجلد)
from engine import (
    ContractRules,
    audit_invoice_dataframe_fast,
    extract_rules_from_contract,
    scan_and_generate_validation_file,
)

# 1. إعدادات الواجهة والنسق البصري
st.set_page_config(
    page_title="Verilogix AI | Audit Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# شريط جانبي احترافي
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/delivery-conveyer.png", width=80
    )
    st.title("Verilogix AI Engine")
    st.caption("منصة التدقيق الذاتي للفواتير والعقود اللوجستية")
    st.divider()
    st.write("👤 **المستخدم:** زين عثمان")
    st.write("🌐 **البيئة:** Cloud Production")

st.title("📦 Verilogix AI | لوحة التحكم والتفتيش اللوجستي")
st.write(
    "أهلاً بك زين! ارفع عقد الشحن وفاتورة الشحنات للبدء في تحليل الفروقات المالية واسترداد المبالغ."
)
st.divider()

UPLOAD_DIR = "server_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 2. قسم رفع المستندات
st.subheader("1️⃣ رفع المستندات المرجعية")
col_up1, col_up2 = st.columns(2)

with col_up1:
    contract_file = st.file_uploader(
        "📄 عقد الشحن المعتمد (PDF)", type=["pdf"], key="pdf_contract"
    )

with col_up2:
    invoice_file = st.file_uploader(
        "📊 فاتورة الشحنات التفصيلية (Excel / CSV)",
        type=["xlsx", "xls", "csv"],
        key="excel_invoice",
    )

st.divider()

# 3. مرحلة استخراج القواعد والتأكيد المسبق
if contract_file and invoice_file:
    contract_path = os.path.join(UPLOAD_DIR, contract_file.name)
    invoice_path = os.path.join(UPLOAD_DIR, invoice_file.name)

    with open(contract_path, "wb") as f:
        f.write(contract_file.getbuffer())
    with open(invoice_path, "wb") as f:
        f.write(invoice_file.getbuffer())

    rules_json_path = os.path.join(
        UPLOAD_DIR, f"{contract_file.name}_rules.json"
    )

    # قراءة أو استخراج قواعد العقد بـ Gemini
    if "contract_rules" not in st.session_state:
        if os.path.exists(rules_json_path):
            with open(rules_json_path, "r", encoding="utf-8") as f:
                st.session_state.contract_rules = ContractRules(**json.load(f))
        else:
            with st.spinner("📜 [Gemini AI] جاري تفكيك ملف العقد واستخراج المصفوفات المالية..."):
                st.session_state.contract_rules = extract_rules_from_contract(
                    contract_path, rules_json_path
                )

    rules = st.session_state.contract_rules

    # عرض ملخص القواعد للتأكيد قبل المعالجة
    st.subheader("2️⃣ مراجعة وتأكيد قواعد العقد (Pre-Audit Approval)")
    st.warning("⚠️ يُرجى مراجعة وتأكيد القواعد التي تم استخراجها من العقد قبل بدء تدقيق الفاتورة:")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("شركة الشحن", rules.carrier_name)
    m2.metric("العملة", rules.currency)
    m3.metric("نسبة الضريبة (VAT)", f"{rules.vat_percentage * 100}%")
    m4.metric("رسوم الوقود", f"{rules.fuel_surcharge_percentage * 100}%")

    with st.expander("🔍 استعراض تفاصيل الشرائح السعرية والرسوم الإضافية", expanded=True):
        st.write("**جدول أسعار المناطق (Rate Cards):**")
        rates_list = [
            {
                "رمز المنطقة": r.zone_id,
                "الوزن الأقصى (كجم)": r.max_weight_kg,
                "السعر الأساسي": f"{r.base_price} {rules.currency}",
                "سعر الكيلو الإضافي": f"{r.extra_kg_price} {rules.currency}",
            }
            for r in rules.rates
        ]
        st.dataframe(pd.DataFrame(rates_list), use_container_width=True)

        st.write("**شروط الرسوم الإضافية (Surcharges):**")
        st.write(f"- **نسبة COD:** {rules.surcharges.cod_fee_percentage * 100}% (بحد أدنى: {rules.surcharges.cod_min_fee} {rules.currency})")
        st.write(f"- **رسوم المناطق النائية:** {rules.surcharges.remote_area_fee} {rules.currency}")
        st.write(f"- **نسبة تكلفة المرتجعات (RTO):** {rules.surcharges.rto_fee_percentage * 100}%")

    st.divider()

    # 4. زر التنفيذ بعد التأكيد
    st.subheader("3️⃣ تشغيل التدقيق الحسابي واللوجستي")

    if st.button("🚀 تأكيد القواعد وبدء معالجة الفاتورة", type="primary", use_container_width=True):
        ext = os.path.splitext(invoice_path)[-1].lower()
        invoice_df = pd.read_csv(invoice_path) if ext == ".csv" else pd.read_excel(invoice_path)

        # أ) المسح الأولي للتأكد من سلامة المدخلات
        validation_file = os.path.join(UPLOAD_DIR, f"Need_Verification_{invoice_file.name}.xlsx")
        has_issues = scan_and_generate_validation_file(
            invoice_df, rules, validation_output_path=validation_file
        )

        if has_issues:
            st.error("🛑 توقف مؤقت: يحتوي ملف الفاتورة على مدن أو مناطق مبهمة تحتاج لتوضيح!")
            with open(validation_file, "rb") as f:
                st.download_button(
                    label="📥 تحميل ملف التوضيحات المطلوب تعبئته",
                    data=f,
                    file_name=os.path.basename(validation_file),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            # ب) تنفيذ التدقيق النهائي
            with st.spinner("⚡ [Verilogix Engine] جاري مطابقة الشحنات واحتساب الفروقات المالية..."):
                flagged_df = audit_invoice_dataframe_fast(invoice_df, rules)

                if flagged_df.empty:
                    st.balloons()
                    st.success("✅ جميع الشحنات مطابقة للعقد والأسعار تماماً دون وجود أي مخالفات أو فروقات!")
                else:
                    st.success("🎉 تم التدقيق بنجاح واكتشاف الفروقات المالية القابلة للاسترداد!")

                    report_path = os.path.join(UPLOAD_DIR, f"Verilogix_Audit_Report_{invoice_file.name}.xlsx")
                    excel_df = flagged_df.rename(
                        columns={
                            "tracking_id": "رقم الشحنة",
                            "actual_weight": "الوزن الفعلي",
                            "volumetric_weight": "الوزن الحجمي",
                            "chargeable_weight": "الوزن المحسوب",
                            "billed_amount": f"المبلغ بالفاتورة ({rules.currency})",
                            "expected_total": f"المبلغ المستحق ({rules.currency})",
                            "overcharge": f"الزيادة المستردة ({rules.currency})",
                            "dispute_evidence": "تقرير النزاع التلقائي",
                        }
                    )
                    excel_df.to_excel(report_path, index=False)

                    total_recovered = excel_df[f"الزيادة المستردة ({rules.currency})"].sum()

                    # عرض البطاقات الرقمية
                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("إجمالي الأموال القابلة للاسترداد", f"{total_recovered:,.2f} {rules.currency}")
                    kpi2.metric("عدد الشحنات المخالفة", len(excel_df))
                    kpi3.metric("نسبة المخالفات بالفاتورة", f"{(len(excel_df) / len(invoice_df)) * 100:.1f}%")

                    st.subheader("📋 تفاصيل النزاعات والمخالفات المكتشفة:")
                    st.dataframe(excel_df, use_container_width=True)

                    # زر تحميل التقرير النهائي
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="📥 تحميل تقرير المطالبة والنزاع المالي (Excel)",
                            data=f,
                            file_name=os.path.basename(report_path),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )
