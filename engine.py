import json
import os
import re
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
    rto_is_addon: bool = Field(
        default=True,
        description="هل رسوم المرتجع تضاف فوق السعر الأساسي أم كنسبة منه فقط",
    )


# نموذج متوافق مع Gemini بدلاً من القاموس المباشر لتجنب خطأ Schema
class CityZoneMapping(BaseModel):
    city: str = Field(description="اسم المدينة أو الرمز البريدي")
    zone_id: str = Field(description="رمز المنطقة المطابق")


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
    weight_unit: str = Field(
        default="kg", description="وحدة الوزن العقدية (kg أو lbs)"
    )
    dim_unit: str = Field(
        default="cm", description="وحدة الأبعاد العقدية (cm أو inches)"
    )
    weight_rounding_increment: float = Field(
        default=0.5,
        description="خطوة تقريب الوزن للأعلى (مثلاً 0.5 كجم أو 1.0 كجم)",
    )
    min_overcharge_threshold: float = Field(
        default=0.5, description="الحد الأدنى لفرق السعر للمطالبة به"
    )
    city_zone_mappings: List[CityZoneMapping] = Field(
        default_factory=list,
        description="قائمة تطابق المدن/الرموز البريدية مع المناطق",
    )
    rates: List[RateZone] = Field(description="جدول الشرائح السعرية والمناطق")
    surcharges: SurchargeRules = Field(default_factory=SurchargeRules)

    # خاصية تحويل قائمة المدن إلى قاموس لضمان توافق باقي الكود الحسابي
    @property
    def city_zone_matrix(self) -> Dict[str, str]:
        return {m.city: m.zone_id for m in self.city_zone_mappings}


# ==========================================
# 2. تنظيف وتحويل البيانات الذكي
# ==========================================


def safe_numeric_conversion(
    series: pd.Series, default_val: float = 0.0
) -> pd.Series:
    """تنظيف ذكي يعالج الفواصل العشرية الأوروبية والعربية والعلامات الخاصة"""

    def clean_val(val):
        if pd.isna(val):
            return default_val
        s = str(val).strip()
        if "," in s and "." in s:
            s = s.replace(",", "")
        elif "," in s and "." not in s:
            s = s.replace(",", ".")
        s = re.sub(r"[^\d.]", "", s)
        try:
            return float(s) if s != "" else default_val
        except ValueError:
            return default_val

    return series.apply(clean_val)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """مطابقة محكمة لأسماء الأعمدة تضمن عدم التداخل بين المبالغ والمستحقات"""
    col_map = {}
    for col in df.columns:
        c_lower = str(col).strip().lower()

        # أولوية المطابقة
        if any(
            k in c_lower
            for k in [
                "tracking",
                "شحنة",
                "awb",
                "waybill",
                "conshipment",
                "barcode",
            ]
        ):
            col_map[col] = "tracking_id"
        elif any(
            k in c_lower for k in ["cod", "دفع عند الاستلام", "cash_on_delivery"]
        ):
            col_map[col] = "cod_amount"
        elif any(
            k in c_lower
            for k in [
                "billed",
                "المبلغ المفلتر",
                "المبلغ بالفاتورة",
                "total_charge",
                "amount_billed",
            ]
        ):
            col_map[col] = "billed_amount"
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
            k in c_lower for k in ["zone", "منطقة", "region", "destination_zone"]
        ):
            col_map[col] = "zone_id"
        elif any(
            k in c_lower for k in ["destination", "المدينة", "city", "postal", "zip"]
        ):
            col_map[col] = "destination_city"
        elif any(k in c_lower for k in ["remote", "نائية", "out_of_delivery"]):
            col_map[col] = "is_remote"
        elif any(
            k in c_lower for k in ["status", "حالة", "rto", "delivered", "returned"]
        ):
            col_map[col] = "shipment_status"
        elif "amount" in c_lower or "cost" in c_lower or "المبلغ" in c_lower:
            if "billed_amount" not in col_map.values():
                col_map[col] = "billed_amount"

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
# 3. الفحص المسبق السريع قبل المعالجة
# ==========================================


def scan_and_generate_validation_file(
    invoice_df: pd.DataFrame,
    rules: ContractRules,
    validation_output_path: str = "Need_User_Verification.xlsx",
) -> bool:
    """فحص سريع ومعدل الأداء للكشف عن البيانات المبهمة"""
    df = normalize_columns(invoice_df.copy())
    valid_zones = {r.zone_id.strip().lower() for r in rules.rates}
    matrix_cities = (
        {k.lower() for k in rules.city_zone_matrix.keys()}
        if rules.city_zone_matrix
        else set()
    )

    issues = []

    for idx, row in df.iterrows():
        tracking = row["tracking_id"]
        zone = str(row["zone_id"]).strip().lower()
        city = str(row["destination_city"]).strip().lower()
        billed = row["billed_amount"]

        reason = []

        if zone not in valid_zones and zone != "default":
            reason.append(f"المنطقة '{row['zone_id']}' غير معرفة بجدول العقد.")

        if matrix_cities and city and city not in matrix_cities:
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

    contract_file = client.files.upload(file=contract_pdf_path)

    prompt = """
    You are an expert global logistics auditor for Verilogix AI. Analyze the shipping contract and extract:
    1. Carrier Name and Currency (USD, EUR, SAR, AED, etc.).
    2. VAT / Tax Percentage.
    3. Fuel Surcharge Percentage.
    4. Volumetric Weight Divisor (e.g. 5000, 6000) and Units (kg/lbs, cm/inches).
    5. Weight Rounding Increment (e.g., 0.5 kg or 1.0 kg rounding steps).
    6. City/Postal Code to Zone Mapping Matrix if present (as city and zone_id list).
    7. All Rate Zones (Max Weight, Base Price, Extra KG Price).
    8. Surcharges: COD %, Minimum COD fee, Remote Area fee, and Return to Origin (RTO) fee percentage & whether RTO is an add-on fee.
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
# 5. المحرك الحسابي والتدقيق الفائق المعدل
# ==========================================


def audit_invoice_dataframe_fast(
    invoice_df: pd.DataFrame, rules: ContractRules
) -> pd.DataFrame:
    df = normalize_columns(invoice_df.copy())

    # أ) تحويل الوحدات القياسية إن وجدت (LBS / INCHES)
    if rules.weight_unit.lower() in ["lb", "lbs", "pound", "باوند"]:
        df["actual_weight_kg"] = df["actual_weight"] * 0.453592
    else:
        df["actual_weight_kg"] = df["actual_weight"]

    if rules.dim_unit.lower() in ["in", "inch", "inches", "إنش", "بوصة"]:
        df["length_cm"] = df["length_cm"] * 2.54
        df["width_cm"] = df["width_cm"] * 2.54
        df["height_cm"] = df["height_cm"] * 2.54

    # ب) تصحيح حساب الوزن الحجمي للشحنات متعددة الطرود (MPS)
    vol_divisor = (
        rules.volumetric_divisor if rules.volumetric_divisor > 0 else 5000.0
    )
    df["piece_volumetric_weight_kg"] = (
        df["length_cm"] * df["width_cm"] * df["height_cm"]
    ) / vol_divisor

    # تجميع الأحجام والأوزان بدقة لكل رقم شحنة
    grouped = df.groupby("tracking_id")[
        ["actual_weight_kg", "piece_volumetric_weight_kg"]
    ].transform("sum")
    df["total_actual_weight_kg"] = grouped["actual_weight_kg"]
    df["total_volumetric_weight_kg"] = grouped["piece_volumetric_weight_kg"]

    df["has_zero_dims"] = (
        (df["length_cm"] == 0) & (df["width_cm"] == 0) & (df["height_cm"] == 0)
    )

    # كشف تكرارات الفاتورة الخاطئة
    df["is_duplicate"] = df.duplicated(
        subset=["tracking_id", "billed_amount"], keep="first"
    ) & ~df["tracking_id"].astype(str).str.startswith("UNKNOWN_")

    # ج) مطابقة المناطق والمدن
    if rules.city_zone_matrix:
        matrix_lower = {k.lower(): v for k, v in rules.city_zone_matrix.items()}
        mapped_zones = (
            df["destination_city"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(matrix_lower)
        )
        df["verified_zone"] = mapped_zones.fillna(df["zone_id"])
        df["zone_mismatch"] = (
            df["verified_zone"].astype(str).str.lower()
            != df["zone_id"].astype(str).str.lower()
        ) & mapped_zones.notna()
    else:
        df["verified_zone"] = df["zone_id"]
        df["zone_mismatch"] = False

    # د) حساب الوزن الخاضع للرسوم مع تطبيق خطوة التقريب للأعلى (Weight Rounding)
    df["raw_chargeable_weight"] = df[
        ["total_actual_weight_kg", "total_volumetric_weight_kg"]
    ].max(axis=1)

    if rules.weight_rounding_increment > 0:
        inc = rules.weight_rounding_increment
        df["chargeable_weight"] = (
            np.ceil(df["raw_chargeable_weight"] / inc) * inc
        )
    else:
        df["chargeable_weight"] = df["raw_chargeable_weight"]

    # هـ) مطابقة الأسعار بناءً على الشرائح
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

    # و) احتساب رسوم المرتجع (RTO)
    is_rto = (
        df["shipment_status"]
        .astype(str)
        .str.lower()
        .isin(["rto", "returned", "مرتجعة", "مرتجع"])
    )

    if rules.surcharges.rto_is_addon:
        df["expected_base_price"] = np.where(
            is_rto,
            df["expected_base_price"]
            * (1.0 + rules.surcharges.rto_fee_percentage),
            df["expected_base_price"],
        )
    else:
        df["expected_base_price"] = np.where(
            is_rto,
            df["expected_base_price"] * rules.surcharges.rto_fee_percentage,
            df["expected_base_price"],
        )

    # ز) احتساب باقي الرسوم والضريبة
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

    # ح) فلترة المخالفات بناءً على الحد المعتمد للعملة
    min_thresh = (
        rules.min_overcharge_threshold
        if rules.min_overcharge_threshold > 0
        else 0.1
    )

    flagged = df[
        (df["overcharge"] >= min_thresh)
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
                "تنبيه: أبعاد الشحنة مفقودة، تم الاحتساب بناءً على الوزن الفعلي فقط."
            )
        if row["overcharge"] >= min_thresh:
            evidence_list.append(
                f"فروقات مالية: المفلتر بالفاتورة {row['billed_amount']} {rules.currency} | المستحق بالعقد {row['expected_total']} {rules.currency}. "
                f"المطالبة باسترداد: {row['overcharge']} {rules.currency}."
            )
        return " | ".join(evidence_list)

    flagged["dispute_evidence"] = flagged.apply(build_evidence, axis=1)

    return flagged[
        [
            "tracking_id",
            "total_actual_weight_kg",
            "total_volumetric_weight_kg",
            "chargeable_weight",
            "billed_amount",
            "expected_total",
            "overcharge",
            "dispute_evidence",
        ]
    ]
    
