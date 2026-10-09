import json
import os
import pandas as pd
import streamlit as st

# استدعاء المحرك الرئيسي المتطور
from engine import (
    ContractRules,
    RateZone,
    SurchargeRules,
    audit_invoice_dataframe_fast,
    extract_rules_from_contract,
    scan_and_generate_validation_file,
)

# 1. إعدادات الواجهة والنسق البصري
st.set_page_config(
    page_title="Verilogix AI | Admin Control Panel",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. الشريط الجانبي لتنظيم العملاء والبيئة
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/delivery-conveyer.png", width=70
    )
    st.title("Verilogix Admin Engine")
    st.caption("منصة التدقيق اللوجستي المتقدمة")
    st.divider()

    client_name = st.text_input(
        "🏢 اسم العميل / المشروع",
        value="Default_Client",
        help="تستعمل لإنشاء مجلد حفظ خاص ببيانات العميل",
    )
    st.write("👤 **المسؤول:** زين عثمان")
    st.write("🌐 **البيئة:** Universal Production")
    st.divider()

st.title("📦 Verilogix AI | لوحة التحكم والتفتيش اللوجستي")
st.write(
    f"أهلاً بك زين! ارفع عقد الشحن وفاتورة الشحنات للبدء في تحليل الفروقات المالية لصالح العميل: **{client_name}**."
)
st.divider()

# مجلد حفظ مخصص للعميل
UPLOAD_DIR = os.path.join("server_uploads", client_name)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 3. قسم رفع المستندات
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

# 4. مرحلة استخراج البيانات والمراجعة التفاعلية قبل التدقيق
if contract_file and invoice_file:
    contract_path = os.path.join(UPLOAD_DIR, contract_file.name)
    invoice_path = os.path.join(UPLOAD_DIR, invoice_file.name)

    # حفظ الملفات المرفوعة
    with open(contract_path, "wb") as f:
        f.write(contract_file.getbuffer())
    with open(invoice_path, "wb") as f:
        f.write(invoice_file.getbuffer())

    rules_json_path = os.path.join(
        UPLOAD_DIR, f"{contract_file.name}_rules.json"
    )

    # استخراج قواعد العقد بـ Gemini عند الرفع لأول مرة
    if (
        "contract_rules" not in st.session_state
        or st.session_state.get("current_contract") != contract_file.name
    ):
        if os.path.exists(rules_json_path):
            with open(rules_json_path, "r", encoding="utf-8") as f:
                st.session_state.contract_rules = ContractRules(**json.load(f))
        else:
            with st.spinner("📜 [Gemini AI] جاري قراءة العقد واستخراج المصفوفات المالية..."):
                st.session_state.contract_rules = extract_rules_from_contract(
                    contract_path, rules_json_path
                )
        st.session_state.current_contract = contract_file.name

    rules: ContractRules = st.session_state.contract_rules

    # ==========================================
    # قسم المراجعة والتعديل المباشر (Editable Form)
    # ==========================================
    st.subheader("2️⃣ مراجعة وتعديل قواعد العقد قبل التدقيق (Pre-Audit Control)")
    st.info(
        "💡 يمكنك مراجعة البيانات التي استخرجها النظام وتعديل أي قيمة قبل بدء معالجة الفاتورة:"
    )

    with st.form("contract_rules_form"):
        # معالجة ذكية لمنع تكرار مضاعفة النسب المئوية في الواجهة (حماية من تحول 5% إلى 500%)
        vat_disp = float(rules.vat_percentage * 100) if rules.vat_percentage <= 1.0 else float(rules.vat_percentage)
        fuel_disp = float(rules.fuel_surcharge_percentage * 100) if rules.fuel_surcharge_percentage <= 1.0 else float(rules.fuel_surcharge_percentage)
        cod_disp = float(rules.surcharges.cod_fee_percentage * 100) if rules.surcharges.cod_fee_percentage <= 1.0 else float(rules.surcharges.cod_fee_percentage)
        rto_disp = float(rules.surcharges.rto_fee_percentage * 100) if rules.surcharges.rto_fee_percentage <= 1.0 else float(rules.surcharges.rto_fee_percentage)

        # أ) الشروط الأساسية والعملات
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            carrier_name = st.text_input("شركة الشحن", rules.carrier_name)
            currency = st.text_input("العملة المعتمدة", rules.currency)
        with col2:
            vat_pct = (
                st.number_input(
                    "نسبة الضريبة VAT %",
                    value=vat_disp,
                    step=0.5,
                )
                / 100.0
            )
            fuel_pct = (
                st.number_input(
                    "نسبة رسوم الوقود %",
                    value=fuel_disp,
                    step=0.5,
                )
                / 100.0
            )
        with col3:
            vol_div = st.number_input(
                "معامل الوزن الحجمي",
                value=float(rules.volumetric_divisor),
                step=100.0,
            )
            round_inc = st.number_input(
                "خطوة تقريب الوزن (كجم)",
                value=float(rules.weight_rounding_increment),
                step=0.1,
            )
        with col4:
            weight_unit = st.selectbox(
                "وحدة الوزن",
                ["kg", "lbs"],
                index=0 if rules.weight_unit.lower() == "kg" else 1,
            )
            min_thresh = st.number_input(
                "حد المطالبة الأدنى",
                value=float(rules.min_overcharge_threshold),
                step=0.1,
            )

        st.divider()

        # ب) الرسوم الإضافية (Surcharges)
        st.write("💳 **الرسوم الإضافية وشرط المرتجعات:**")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            cod_pct = (
                st.number_input(
                    "نسبة COD %",
                    value=cod_disp,
                    step=0.5,
                )
                / 100.0
            )
        with sc2:
            cod_min = st.number_input(
                "الحد الأدنى لرسوم COD",
                value=float(rules.surcharges.cod_min_fee),
                step=1.0,
            )
        with sc3:
            remote_fee = st.number_input(
                "رسوم المناطق النائية",
                value=float(rules.surcharges.remote_area_fee),
                step=1.0,
            )
        with sc4:
            rto_pct = (
                st.number_input(
                    "نسبة المرتجع RTO %",
                    value=rto_disp,
                    step=5.0,
                )
                / 100.0
            )
            rto_addon = st.checkbox(
                "رسوم المرتجع إضافية فوق السعر الأساسي",
                value=rules.surcharges.rto_is_addon,
            )

        st.divider()

        # ج) جدول الشرائح السعرية والمناطق القابل للتعديل المباشر
        st.write("📊 **شرائح الأسعار والمناطق (Rate Cards):**")
        rates_data = [
            {
                "zone_id": r.zone_id,
                "max_weight_kg": r.max_weight_kg,
                "base_price": r.base_price,
                "extra_kg_price": r.extra_kg_price,
            }
            for r in rules.rates
        ]
        rates_df = pd.DataFrame(rates_data)

        edited_rates_df = st.data_editor(
            rates_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "zone_id": "رمز المنطقة",
                "max_weight_kg": "الوزن الأقصى (كجم)",
                "base_price": f"السعر الأساسي ({currency})",
                "extra_kg_price": f"الكيلو الإضافي ({currency})",
            },
        )

        save_rules_btn = st.form_submit_button(
            "💾 حفظ وتحديث القواعد المعتمدة"
        )

        # تحديث كائن القواعد عند الضغط على زر الحفظ داخل النموذج
        if save_rules_btn:
            new_rates = [
                RateZone(
                    zone_id=row["zone_id"],
                    max_weight_kg=float(row["max_weight_kg"]),
                    base_price=float(row["base_price"]),
                    extra_kg_price=float(row["extra_kg_price"]),
                )
                for _, row in edited_rates_df.iterrows()
            ]

            st.session_state.contract_rules = ContractRules(
                carrier_name=carrier_name,
                currency=currency,
                vat_percentage=vat_pct,
                fuel_surcharge_percentage=fuel_pct,
                volumetric_divisor=vol_div,
                weight_unit=weight_unit,
                weight_rounding_increment=round_inc,
                min_overcharge_threshold=min_thresh,
                city_zone_matrix=rules.city_zone_matrix,
                rates=new_rates,
                surcharges=SurchargeRules(
                    cod_fee_percentage=cod_pct,
                    cod_min_fee=cod_min,
                    remote_area_fee=remote_fee,
                    rto_fee_percentage=rto_pct,
                    rto_is_addon=rto_addon,
                ),
            )

            # حفظ القواعد المعدلة في JSON العميل
            with open(rules_json_path, "w", encoding="utf-8") as f:
                f.write(
                    st.session_state.contract_rules.model_dump_json(indent=4)
                )

            st.success("✅ تم تحديث قواعد العقد بنجاح!")
            st.rerun()

    st.divider()

    # 5. تشغيل التدقيق الحسابي بعد التأكيد
    st.subheader("3️⃣ تشغيل المعالجة والتدقيق")

    if st.button(
        "🚀 بدء تدقيق الفاتورة الآن",
        type="primary",
        use_container_width=True,
    ):
        ext = os.path.splitext(invoice_path)[-1].lower()
        invoice_df = (
            pd.read_csv(invoice_path)
            if ext == ".csv"
            else pd.read_excel(invoice_path)
        )

        validation_file = os.path.join(
            UPLOAD_DIR, f"Need_Verification_{invoice_file.name}.xlsx"
        )
        
        # الفحص المرن الفائق
        has_issues = scan_and_generate_validation_file(
            invoice_df,
            st.session_state.contract_rules,
            validation_output_path=validation_file,
            strict_mode=False,
        )

        if has_issues:
            st.error("🛑 توقف مؤقت: ملف الفاتورة فارغ أو يحتوي على بيانات غير صحيحة.")
        else:
            with st.spinner("⚡ [Verilogix Engine] جاري مطابقة الشحنات واستخراج الفروقات..."):
                full_audit_df = audit_invoice_dataframe_fast(
                    invoice_df, st.session_state.contract_rules
                )

                # فلترة الشحنات التي تحتوي على مخالفات/ملاحظات للجدول
                flagged_df = full_audit_df[
                    full_audit_df["dispute_evidence"] != "مطابق للعقد"
                ].copy()

                if flagged_df.empty:
                    st.balloons()
                    st.success(
                        "✅ جميع الشحنات مطابقة للعقد تماماً دون وجود أي مخالفات أو فروقات مالية!"
                    )
                else:
                    st.success("🎉 تم التدقيق بنجاح واستخراج تحليل الفروقات الماليّة!")

                    report_path = os.path.join(
                        UPLOAD_DIR,
                        f"Verilogix_Audit_Report_{invoice_file.name}.xlsx",
                    )
                    curr = st.session_state.contract_rules.currency

                    excel_df = flagged_df.rename(
                        columns={
                            "tracking_id": "رقم الشحنة",
                            "total_actual_weight_kg": "الوزن الفعلي (كجم)",
                            "total_volumetric_weight_kg": "الوزن الحجمي (كجم)",
                            "chargeable_weight": "الوزن المحسوب (كجم)",
                            "billed_amount": f"المبلغ بالفاتورة ({curr})",
                            "expected_total": f"المبلغ المستحق ({curr})",
                            "overcharge": f"الفروقات المالية ({curr})",
                            "dispute_evidence": "تقرير النزاع التلقائي",
                        }
                    )
                    excel_df.to_excel(report_path, index=False)

                    # حساب إجمالي الزيادات الصريحة (Gross Overcharges)
                    gross_overcharges = full_audit_df[
                        full_audit_df["overcharge"] > 0
                    ]["overcharge"].sum()

                    # حساب صافي التسوية المالية الاستردادية (Net Settlement)
                    total_billed = full_audit_df["billed_amount"].sum()
                    total_expected = full_audit_df["expected_total"].sum()
                    net_settlement = total_billed - total_expected

                    # عرض البطاقات الرقمية
                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric(
                        "إجمالي الزيادات القابلة للاسترداد (Gross)",
                        f"{gross_overcharges:,.2f} {curr}",
                    )
                    kpi2.metric(
                        "صافي التسوية والاسترداد (Net)",
                        f"{net_settlement:,.2f} {curr}",
                        help="الفرق الصافي الكامل بين إجمالي الفاتورة وإجمالي المستحق الفعلي بالعقد بعد المقاصة",
                    )
                    kpi3.metric("عدد الشحنات المخالفة", len(excel_df))
                    kpi4.metric(
                        "نسبة المخالفات بالفاتورة",
                        f"{(len(excel_df) / len(invoice_df)) * 100:.1f}%",
                    )

                    st.subheader("📋 جدول النزاعات والمخالفات المكتشفة:")
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
    
