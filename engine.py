import json
import os
from typing import Dict, List, Optional
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

# إعداد العميل
client = genai.Client()

# ==========================================
# 1. هياكل البيانات العالمية (Global Models)
# ==========================================


class RateZone(BaseModel):
    zone_id: str = Field(
        description="رمز المنطقة أو الفئة عالمياً، مثال: Zone_A, EU_Intra, Domestic"
    )
    max_weight_kg: float = Field(description="الحد الأقصى للوزن بالكيلوجرام")
    base_price: float = Field(
        description="السعر الأساسي المعتمد بالعملة المحلية"
    )
    extra_kg_price: float = Field(default=0.0, description="سعر الكيلو الإضافي")


class SurchargeRules(BaseModel):
    cod_fee_percentage: float = Field(
        default=0.0, description="نسبة رسوم الدفع عند الاستلام"
    )
    cod_min_fee: float = Field(default=0.0, description="الحد الأدنى لرسوم COD")
    remote_area_fee: float = Field(
        default=0.0, description="رسوم المناطق النائية الثابتة"
    )
    rto_fee_percentage: float = Field(
        default=1.0,
        description="نسبة تكلفة الشحنة المرتجعة مقارنة بالسعر الأساسي",
    )


class ContractRules(BaseModel):
    carrier_name: str = Field(description="اسم شركة الشحن العالمية أو المحلية")
    currency: str = Field(
        default="USD",
        description="رمز العملة الخاص بالعقد (USD, EUR, SAR, AED...)",
    )
    vat_percentage: float = Field(
        default=0.0, description="نسبة الضريبة المعمول بها في دولة العقد"
    )
    fuel_surcharge_percentage: float = Field(
        default=0.0, description="نسبة رسوم الوقود المتفق عليها"
    )
    volumetric_divisor: float = Field(
        default=5000.0, description="معامل الوزن الحجمي (5000 أو 6000 أو custom)"
    )
    city_zone_matrix: Dict[str, str] = Field(
        default_factory=dict,
        description="مصفوفة تطابق المدن/الرموز البريدية مع المناطق",
    )
    rates: List[RateZone] = Field(description="جدول الشرائح السعرية والمناطق")
    surcharges: SurchargeRules = Field(default_factory=SurchargeRules)


# ==========================================
# 2. تنظيف وتحويل البيانات
# ==========================================


def safe_numeric_conversion(series: pd.Series, default_val: float = 0.0) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", str(default_val))
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(default_val)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        c_lower = str(col).strip().lower()
        if any(
            k in c_lower
            for k in ["tracking", "شحنة", "awb", "waybill", "conshipment", "barcode"]
        ):
            col_map[col] = "tracking_id"
        elif any(
            k in c_lower
            for k in ["actual_weight", "وزن", "weight", "net_weight", "gross_weight"]
        ):
            col_map[col] = "actual_weight"
        elif any(k in c_lower for k in ["length", "طول", "l_cm"]):
            col_map[col] = "length_cm"
        elif any(k in c_lower for k in ["width", "عرض", "w_cm"]):
            col_map[col] = "width_cm"
        elif any(k in c_lower for k in ["height", "ارتفاع", "h_cm"]):
            col_map[col] = "height_cm"
        elif any(
            k in c_lower
            for k in ["billed", "المبلغ", "amount", "charge", "total_fee", "cost"]
        ):
            col_map[col] = "billed_amount"
        elif any(
            k in c_lower for k in ["zone", "منطقة", "region", "destination_zone"]
        ):
            col_map[col] = "zone_id"
        elif any(
            k in c_lower for k in ["destination", "المدينة", "city", "postal", "zip"]
        ):
            col_map[col] = "destination_city"
        elif any(
            k in c_lower for k in ["cod", "دفع عند الاستلام", "cash_on_delivery"]
        ):
            col_map[col] = "cod_amount"
        elif any(k in c_lower for k in ["remote", "نائية", "out_of_delivery"]):
            col_map[col] = "is_remote"
        elif any(
            k in c_lower for k in ["status", "حالة", "rto", "delivered", "returned"]
        ):
            col_map[col] = "shipment_status"

    df = df.rename(columns=col_map)

    numeric_cols = [
        "actual_weight",
        "length_cm",
        "width_cm",
        "height_cm",
        "billed_amount",
        "cod_amount",
    ]
    for n_col in numeric_cols:
        if n_col not in df.columns:
            df[n_col] = 0.0
        else:
            df[n_col] = safe_numeric_conversion(df[n_col])

    if "tracking_id" not in df.columns:
        df["tracking_id"] = [f"UNKNOWN_{i}" for i in range(len(df))]

    if "destination_city" not in df.columns:
        df["destination_city"] = ""

    if "is_remote" not in df.columns:
        df["is_remote"] = False
    else:
        df["is_remote"] = (
            df["is_remote"].astype(str).str.lower().isin(["true", "1", "نعم", "yes"])
        )

    if "shipment_status" not in df.columns:
        df["shipment_status"] = "delivered"

    if "zone_id" not in df.columns:
        df["zone_id"] = "default"

    return df


# ==========================================
# 3. ميزة المسح الأولي واستخراج ملف التوضيحات للعميل
# ==========================================


def scan_and_generate_validation_file(
    invoice_df: pd.DataFrame,
    rules: ContractRules,
    validation_output_path: str = "Need_User_Verification.xlsx",
) -> bool:
    """فحص الفاتورة قبل المعالجة وتوليد ملف إكسل بالتوضيحات المطلوبة إن وجدت"""
    df = normalize_columns(invoice_df.copy())
    valid_zones = [r.zone_id.strip().lower() for r in rules.rates]

    issues = []

    for idx, row in df.iterrows():
        tracking = row["tracking_id"]
        zone = str(row["zone_id"]).strip().lower()
        city = str(row["destination_city"]).strip().lower()
        billed = row["billed_amount"]

        reason = []

        if zone not in valid_zones and zone != "default":
            reason.append(f"المنطقة '{row['zone_id']}' غير معرفة بجدول العقد.")

        if (
            rules.city_zone_matrix
            and city
            and city not in [k.lower() for k in rules.city_zone_matrix.keys()]
        ):
            reason.append(f"المدينة '{row['destination_city']}' غير مسجلة بالماتريكس.")

        if billed <= 0:
            reason.append("المبلغ المفلتر بالفاتورة يساوي 0 أو غير صحيح.")

        if reason:
            issues.append(
                {
                    "رقم الشحنة": tracking,
                    "المدينة بالمدخلات": row["destination_city"],
                    "المنطقة بالمدخلات": row["zone_id"],
                    "المبلغ بالفاتورة": billed,
                    "سبب التوقف والمراجعة": " | ".join(reason),
                    "المنطقة الصحيحة (تعبئة المستخدم)": "",
                }
            )

    if issues:
        issues_df = pd.DataFrame(issues)
        issues_df.to_excel(validation_output_path, index=False)
        print("\n" + "⚠️ " * 15)
        print("🛑 [توقف مؤقت]: تم العثور على بيانات مبهمة تحتاج لتوضيح العميل.")
        print(f"📁 تم حفظ ملف التوضيحات المطلوب تعبئته في: {validation_output_path}")
        print("👉 يرجى مراجعة الملف وتحديد المناطق الصحيحة ثم إعادة تشغيل التدقيق.")
        print("⚠️ " * 15 + "\n")
        return True

    return False


# ==========================================
# 4. قراءة العقود عبر Gemini
# ==========================================


def extract_rules_from_contract(
    contract_pdf_path: str, output_json_path: str = "contract_rules.json"
) -> ContractRules:
    if not os.path.exists(contract_pdf_path):
        raise FileNotFoundError(f"❌ لم يتم العثور على العقد: {contract_pdf_path}")

    print("📜 [Verilogix AI Global Engine] جاري تحليل العقد وقراءة القواعد...")
    contract_file = client.files.upload(file=contract_pdf_path)

    prompt = """
    You are an expert global logistics auditor for Verilogix AI. Analyze the shipping contract and extract:
    1. Carrier Name and Currency (USD, EUR, SAR, etc.).
    2. VAT / Tax Percentage.
    3. Fuel Surcharge Percentage.
    4. Volumetric Weight Divisor (e.g. 5000, 6000).
    5. City/Postal Code to Zone Mapping Matrix if present.
    6. All Rate Zones (Max Weight, Base Price, Extra KG Price).
    7. Surcharges: COD %, Minimum COD fee, Remote Area fee, and Return to Origin (RTO) fee percentage.
    """

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[contract_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ContractRules,
                temperature=0.0,
            ),
        )
        rules_dict = json.loads(response.text)
        rules = ContractRules(**rules_dict)

        with open(output_json_path, "w", encoding="utf-8") as f:
            f.write(rules.model_dump_json(indent=4))

        return rules
    finally:
        client.files.delete(name=contract_file.name)


# ==========================================
# 5. عرض الشروط واستلام التوافق من العميل
# ==========================================


def confirm_contract_rules_with_user(rules: ContractRules) -> bool:
    """عرض الشروط المستخرجة من العقد واشتراط موافقة المستخدم قبل التدقيق"""
    print("\n" + "=" * 60)
    print("📜 [Verilogix AI] الشروط والقواعد المعتمدة المستخرجة من العقد:")
    print("=" * 60)
    print(f"🏢 شركة الشحن: {rules.carrier_name}")
    print(f"💰 العملة المعتمدة: {rules.currency}")
    print(f"٪ ضريبة القيمة المضافة (VAT): {rules.vat_percentage * 100}%")
    print(f"⛽ رسوم الوقود: {rules.fuel_surcharge_percentage * 100}%")
    print(f"📦 معامل الوزن الحجمي: {rules.volumetric_divisor}")
    print("\n📊 الشرائح السعرية والمناطق:")
    for rate in rules.rates:
        print(
            f"   - المنطقة: {rate.zone_id:<12} | حد الوزن: {rate.max_weight_kg} كجم | السعر: {rate.base_price} {rules.currency} (الكيلو الإضافي: {rate.extra_kg_price})"
        )

    print("\n💳 الرسوم الإضافية (Surcharges):")
    print(
        f"   - نسبة COD: {rules.surcharges.cod_fee_percentage * 100}% (الحد الأدنى: {rules.surcharges.cod_min_fee} {rules.currency})"
    )
    print(
        f"   - رسوم المناطق النائية: {rules.surcharges.remote_area_fee} {rules.currency}"
    )
    print(
        f"   - نسبة الشحنات المرتجعة RTO: {rules.surcharges.rto_fee_percentage * 100}%"
    )
    print("=" * 60 + "\n")

    user_input = (
        input(
            "❓ هل تؤكد صحة هذه البيانات للبدء في تدقيق الفاتورة؟ (اكتب 'yes' للبدء / أو 'no' للتعديل): "
        )
        .strip()
        .lower()
    )

    return user_input in ["yes", "y", "نعم"]


# ==========================================
# 6. المحرك الحسابي والتدقيق الشامل
# ==========================================


def audit_invoice_dataframe_fast(
    invoice_df: pd.DataFrame, rules: ContractRules
) -> pd.DataFrame:
    print("⚡ [Verilogix AI] جاري تنفيذ التدقيق الحسابي واللوجستي الفائق...")

    df = normalize_columns(invoice_df.copy())

    df["has_zero_dims"] = (
        (df["length_cm"] == 0) & (df["width_cm"] == 0) & (df["height_cm"] == 0)
    )
    grouped_weights = df.groupby("tracking_id")[
        ["actual_weight", "length_cm", "width_cm", "height_cm"]
    ].transform("sum")
    df["actual_weight"] = np.where(
        df.duplicated("tracking_id", keep=False),
        grouped_weights["actual_weight"],
        df["actual_weight"],
    )

    df["is_duplicate"] = df.duplicated(
        subset=["tracking_id", "billed_amount"], keep="first"
    ) & ~df["tracking_id"].astype(str).str.startswith("UNKNOWN_")

    if rules.city_zone_matrix:
        mapped_zones = (
            df["destination_city"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({k.lower(): v for k, v in rules.city_zone_matrix.items()})
        )
        df["verified_zone"] = mapped_zones.fillna(df["zone_id"])
        df["zone_mismatch"] = (
            df["verified_zone"].astype(str).str.lower()
            != df["zone_id"].astype(str).str.lower()
        ) & mapped_zones.notna()
    else:
        df["verified_zone"] = df["zone_id"]
        df["zone_mismatch"] = False

    vol_divisor = (
        rules.volumetric_divisor if rules.volumetric_divisor > 0 else 5000.0
    )
    df["volumetric_weight"] = (
        df["length_cm"] * df["width_cm"] * df["height_cm"]
    ) / vol_divisor
    df["chargeable_weight"] = df[["actual_weight", "volumetric_weight"]].max(
        axis=1
    )

    rates_data = [
        {
            "zone_id": r.zone_id.strip().lower(),
            "max_weight_kg": r.max_weight_kg,
            "base_price": r.base_price,
            "extra_kg_price": r.extra_kg_price,
        }
        for r in rules.rates
    ]
    rates_df = pd.DataFrame(rates_data)

    prices = np.zeros(len(df))
    for zone, group_indices in df.groupby(
        df["verified_zone"].astype(str).str.strip().str.lower()
    ).groups.items():
        zone_rates = rates_df[rates_df["zone_id"] == zone]
        if zone_rates.empty:
            zone_rates = rates_df[rates_df["zone_id"] == "default"]
            if zone_rates.empty:
                zone_rates = rates_df

        zone_rates = zone_rates.sort_values("max_weight_kg")
        weights = df.loc[group_indices, "chargeable_weight"].values
        group_prices = np.zeros(len(weights))

        for _, rate in zone_rates.iterrows():
            mask = weights <= rate["max_weight_kg"]
            group_prices = np.where(
                (group_prices == 0) & mask, rate["base_price"], group_prices
            )

        max_rate = zone_rates.iloc[-1]
        over_mask = group_prices == 0
        extra_w = np.maximum(0, weights - max_rate["max_weight_kg"])
        group_prices = np.where(
            over_mask,
            max_rate["base_price"] + (extra_w * max_rate["extra_kg_price"]),
            group_prices,
        )

        prices[group_indices] = group_prices

    df["expected_base_price"] = prices

    is_rto = (
        df["shipment_status"]
        .astype(str)
        .str.lower()
        .isin(["rto", "returned", "مرتجعة", "مرتجع"])
    )
    df["expected_base_price"] = np.where(
        is_rto,
        df["expected_base_price"] * rules.surcharges.rto_fee_percentage,
        df["expected_base_price"],
    )

    df["expected_fuel_fee"] = (
        df["expected_base_price"] * rules.fuel_surcharge_percentage
    )
    df["expected_cod_fee"] = np.where(
        (df["cod_amount"] > 0) & (~is_rto),
        np.maximum(
            df["cod_amount"] * rules.surcharges.cod_fee_percentage,
            rules.surcharges.cod_min_fee,
        ),
        0.0,
    )
    df["expected_remote_fee"] = np.where(
        df["is_remote"], rules.surcharges.remote_area_fee, 0.0
    )

    df["expected_subtotal"] = (
        df["expected_base_price"]
        + df["expected_fuel_fee"]
        + df["expected_cod_fee"]
        + df["expected_remote_fee"]
    )
    df["expected_vat"] = df["expected_subtotal"] * rules.vat_percentage
    df["expected_total"] = (df["expected_subtotal"] + df["expected_vat"]).round(2)

    df["overcharge"] = (df["billed_amount"] - df["expected_total"]).round(2)

    flagged = df[
        (df["overcharge"] > 0.5)
        | (df["is_duplicate"])
        | (df["zone_mismatch"])
        | (df["has_zero_dims"])
    ].copy()

    def build_evidence(row):
        evidence_list = []
        if row["is_duplicate"]:
            evidence_list.append(f"تكرار فاتورة لرقم الشحنة {row['tracking_id']}.")
        if row["zone_mismatch"]:
            evidence_list.append(
                f"تلاعب بالمناطق: الفاتورة سجلت ({row['zone_id']}) بينما الوجهة الفعليّة هي ({row['verified_zone']})."
            )
        if row["has_zero_dims"]:
            evidence_list.append(
                "تنبيه: أبعاد الشحنة مفقودة (0x0x0)، تم الاحتساب بناءً على الوزن الفعلي فقط."
            )
        if row["overcharge"] > 0.5:
            evidence_list.append(
                f"فروقات مالية: المفلتر بالفاتورة {row['billed_amount']} {rules.currency} | المستحق بالعقد {row['expected_total']} {rules.currency}. "
                f"المطالبة باسترداد: {row['overcharge']} {rules.currency}."
            )
        return " | ".join(evidence_list)

    flagged["dispute_evidence"] = flagged.apply(build_evidence, axis=1)

    return flagged[
        [
            "tracking_id",
            "actual_weight",
            "volumetric_weight",
            "chargeable_weight",
            "billed_amount",
            "expected_total",
            "overcharge",
            "dispute_evidence",
        ]
    ]


# ==========================================
# 7. خط التشغيل الرئيسي مع التوقف والتأكيد
# ==========================================


def run_verilogix_global_pipeline(
    contract_pdf_path: str,
    invoice_file_path: str,
    output_report_path: str = "Verilogix_Global_Audit_Report.xlsx",
    rules_json_path: str = "contract_rules.json",
):
    if os.path.exists(rules_json_path):
        print(
            f"📂 [Verilogix] تحميل قواعد العقد المعتمدة ({rules_json_path})..."
        )
        with open(rules_json_path, "r", encoding="utf-8") as f:
            rules = ContractRules(**json.load(f))
    else:
        rules = extract_rules_from_contract(contract_pdf_path, rules_json_path)

    # توقف واشتراط تأكيد المستخدم أولاً
    is_confirmed = confirm_contract_rules_with_user(rules)

    if not is_confirmed:
        print("\n🛑 [تم إيقاف العملية]: لم يتم تأكيد قواعد العقد.")
        print(
            f"👉 يمكنك مراجعة وتعديل الملف '{rules_json_path}' يدوياً ثم إعادة تشغيل الكود.\n"
        )
        return

    print("\n🚀 تم التأكيد! جاري البدء بتدقيق الفاتورة الآن...\n")

    ext = os.path.splitext(invoice_file_path)[-1].lower()
    if ext == ".csv":
        invoice_df = pd.read_csv(invoice_file_path)
    else:
        invoice_df = pd.read_excel(invoice_file_path)

    has_issues = scan_and_generate_validation_file(
        invoice_df, rules, validation_output_path="Need_User_Verification.xlsx"
    )

    if has_issues:
        return

    flagged_df = audit_invoice_dataframe_fast(invoice_df, rules)

    if flagged_df.empty:
        print("✅ جميع الشحنات مطابقة للعقد والأسعار تماماً دون وجود أي مخالفات!")
        return

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

    excel_df.to_excel(output_report_path, index=False)
    total_recovered = excel_df[f"الزيادة المستردة ({rules.currency})"].sum()

    print("\n" + "=" * 60)
    print("🌍 تم التدقيق اللوجستي بنجاح بواسطة Verilogix AI Engine!")
    print(f"🏢 شركة الشحن: {rules.carrier_name}")
    print(f"📊 إجمالي المخالفات المكتشفة: {len(excel_df)}")
    print(
        f"💰 إجمالي المبالغ القابلة للاسترداد: {total_recovered:,.2f} {rules.currency}"
    )
    print(f"📁 تم حفظ التقرير الشامل في: {output_report_path}")
    print("=" * 60)


if __name__ == "__main__":
    pass
