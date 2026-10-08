import json
import os
import re
from typing import Dict, List, Optional
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

client = genai.Client()

# ==========================================
# 1. هياكل البيانات العالمية (Global Models)
# ==========================================

class RateZone(BaseModel):
    zone_id: str = Field(description="رمز المنطقة أو الفئة عالمياً")
    max_weight_kg: float = Field(description="الحد الأقصى للوزن بالكيلوجرام")
    base_price: float = Field(description="السعر الأساسي المعتمد بالعملة المحلية")
    extra_kg_price: float = Field(default=0.0, description="سعر الكيلو الإضافي")

class SurchargeRules(BaseModel):
    cod_fee_percentage: float = Field(default=0.0, description="نسبة رسوم الدفع عند الاستلام")
    cod_min_fee: float = Field(default=0.0, description="الحد الأدنى لرسوم COD")
    remote_area_fee: float = Field(default=0.0, description="رسوم المناطق النائية الثابتة")
    rto_fee_percentage: float = Field(default=1.0, description="نسبة تكلفة الشحنة المرتجعة")
    rto_is_addon: bool = Field(default=True, description="هل رسوم المرتجع تضاف فوق السعر الأساسي")

class CityZoneMapping(BaseModel):
    city: str = Field(description="اسم المدينة أو الرمز البريدي")
    zone_id: str = Field(description="رمز المنطقة المطابق")

class ContractRules(BaseModel):
    carrier_name: str = Field(description="اسم شركة الشحن")
    currency: str = Field(default="USD", description="رمز العملة")
    vat_percentage: float = Field(default=0.0, description="نسبة الضريبة")
    fuel_surcharge_percentage: float = Field(default=0.0, description="نسبة رسوم الوقود")
    volumetric_divisor: float = Field(default=5000.0, description="معامل الوزن الحجمي")
    weight_unit: str = Field(default="kg", description="وحدة الوزن")
    dim_unit: str = Field(default="cm", description="وحدة الأبعاد")
    weight_rounding_increment: float = Field(default=0.5, description="خطوة تقريب الوزن")
    min_overcharge_threshold: float = Field(default=0.1, description="الحد الأدنى للمطالبة")
    city_zone_mappings: List[CityZoneMapping] = Field(default_factory=list)
    rates: List[RateZone] = Field(description="جدول الشرائح السعرية والمناطق")
    surcharges: SurchargeRules = Field(default_factory=SurchargeRules)

    @property
    def city_zone_matrix(self) -> Dict[str, str]:
        return {m.city: m.zone_id for m in self.city_zone_mappings}

# ==========================================
# 2. تنظيف وتحويل البيانات المحصن
# ==========================================

def safe_numeric_conversion(series: pd.Series, default_val: float = 0.0) -> pd.Series:
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

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    return series.apply(clean_val)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()

    col_map = {}
    for col in df.columns:
        c_lower = str(col).strip().lower()

        if any(k in c_lower for k in ["tracking", "شحنة", "awb", "waybill", "conshipment", "barcode"]):
            col_map[col] = "tracking_id"
        elif any(k in c_lower for k in ["cod", "دفع عند الاستلام", "cash_on_delivery"]):
            col_map[col] = "cod_amount"
        elif any(k in c_lower for k in ["billed", "المبلغ المفلتر", "المبلغ بالفاتورة", "total_charge", "amount_billed"]):
            col_map[col] = "billed_amount"
        elif any(k in c_lower for k in ["actual_weight", "وزن", "weight", "net_weight", "gross_weight"]):
            col_map[col] = "actual_weight"
        elif any(k in c_lower for k in ["length", "طول", "l_cm"]):
            col_map[col] = "length_cm"
        elif any(k in c_lower for k in ["width", "عرض", "w_cm"]):
            col_map[col] = "width_cm"
        elif any(k in c_lower for k in ["height", "ارتفاع", "h_cm"]):
            col_map[col] = "height_cm"
        elif any(k in c_lower for k in ["zone", "منطقة", "region", "destination_zone"]):
            col_map[col] = "zone_id"
        elif any(k in c_lower for k in ["destination", "المدينة", "city", "postal", "zip"]):
            col_map[col] = "destination_city"
        elif any(k in c_lower for k in ["remote", "نائية", "out_of_delivery"]):
            col_map[col] = "is_remote"
        elif any(k in c_lower for k in ["status", "حالة", "rto", "delivered", "returned"]):
            col_map[col] = "shipment_status"

    df = df.rename(columns=col_map)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    numeric_cols = ["actual_weight", "length_cm", "width_cm", "height_cm", "billed_amount", "cod_amount"]
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
        df["is_remote"] = df["is_remote"].astype(str).str.lower().isin(["true", "1", "نعم", "yes"])

    if "shipment_status" not in df.columns:
        df["shipment_status"] = "delivered"

    if "zone_id" not in df.columns:
        df["zone_id"] = "default"

    return df

# ==========================================
# 3. الفحص السلس المرن (Fault-Tolerant Engine)
# ==========================================

def scan_and_generate_validation_file(
    invoice_df: pd.DataFrame,
    rules: ContractRules,
    validation_output_path: str = "Need_User_Verification.xlsx",
    strict_mode: bool = False,
) -> bool:
    df = normalize_columns(invoice_df.copy())
    
    if df.empty or "billed_amount" not in df.columns:
        issues_df = pd.DataFrame([{"سبب التوقف": "الملف فارغ أو لا يحتوي على مبالغ شحن مالية."}])
        issues_df.to_excel(validation_output_path, index=False)
        return True

    if strict_mode:
        valid_zones = {r.zone_id.strip().lower() for r in rules.rates}
        matrix_cities = {k.lower(): v.lower() for k, v in rules.city_zone_matrix.items()} if rules.city_zone_matrix else {}
        issues = []

        for _, row in df.iterrows():
            tracking = str(row["tracking_id"]).strip()
            zone = str(row["zone_id"]).strip()
            city = str(row["destination_city"]).strip()
            billed = float(row["billed_amount"])
            reason = []

            zone_is_valid = zone.lower() in valid_zones or zone.lower() == "default"
            if matrix_cities and not matrix_cities.get(city.lower()) and not zone_is_valid:
                reason.append(f"المدينة '{city}' والمنطقة '{zone}' غير معرفتين بجدول العقد.")
            elif not matrix_cities and not zone_is_valid:
                reason.append(f"المنطقة '{zone}' غير معرفة بجدول العقد.")

            if reason:
                issues.append({
                    "رقم الشحنة": tracking,
                    "المدينة": city,
                    "المنطقة": zone,
                    "المبلغ": billed,
                    "سبب التوقف": " | ".join(reason),
                })

        if issues:
            pd.DataFrame(issues).to_excel(validation_output_path, index=False)
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
    6. City/Postal Code to Zone Mapping Matrix if present.
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
# 5. المحرك الحسابي التدقيقي الصارم والمحكم
# ==========================================

def audit_invoice_dataframe_fast(
    invoice_df: pd.DataFrame, rules: ContractRules
) -> pd.DataFrame:
    df = normalize_columns(invoice_df.copy())

    # كشف التكرار المطابق 100% (Duplicate Billing Detection)
    df["is_duplicate"] = df.duplicated(
        subset=["tracking_id", "billed_amount", "actual_weight", "length_cm", "width_cm", "height_cm"],
        keep="first"
    ) & ~df["tracking_id"].astype(str).str.startswith("UNKNOWN_")

    # أ) تحويل الوحدات القياسية (LBS / INCHES)
    if rules.weight_unit.lower() in ["lb", "lbs", "pound", "باوند"]:
        df["actual_weight_kg"] = df["actual_weight"] * 0.453592
    else:
        df["actual_weight_kg"] = df["actual_weight"]

    if rules.dim_unit.lower() in ["in", "inch", "inches", "إنش", "بوصة"]:
        df["length_cm"] = df["length_cm"] * 2.54
        df["width_cm"] = df["width_cm"] * 2.54
        df["height_cm"] = df["height_cm"] * 2.54

    # ب) حساب الوزن الحجمي
    vol_divisor = rules.volumetric_divisor if rules.volumetric_divisor > 0 else 5000.0
    df["piece_volumetric_weight_kg"] = (df["length_cm"] * df["width_cm"] * df["height_cm"]) / vol_divisor

    # تجميع أوزان الطرود للشحنات غير المكررة فقط
    non_dup_mask = ~df["is_duplicate"]
    tracking_col = df.loc[non_dup_mask, "tracking_id"]
    if isinstance(tracking_col, pd.DataFrame):
        tracking_col = tracking_col.iloc[:, 0]

    grouped_act = df[non_dup_mask].groupby(tracking_col)["actual_weight_kg"].transform("sum")
    grouped_vol = df[non_dup_mask].groupby(tracking_col)["piece_volumetric_weight_kg"].transform("sum")

    df["total_actual_weight_kg"] = df["actual_weight_kg"]
    df.loc[non_dup_mask, "total_actual_weight_kg"] = grouped_act

    df["total_volumetric_weight_kg"] = df["piece_volumetric_weight_kg"]
    df.loc[non_dup_mask, "total_volumetric_weight_kg"] = grouped_vol

    df["has_zero_dims"] = (df["length_cm"] == 0) & (df["width_cm"] == 0) & (df["height_cm"] == 0)

    # ج) مطابقة المناطق والمدن الذكية
    valid_zones = {r.zone_id.strip().lower() for r in rules.rates}
    default_zone = list(valid_zones)[0] if valid_zones else "standard"

    if rules.city_zone_matrix:
        matrix_lower = {k.lower(): v.strip().lower() for k, v in rules.city_zone_matrix.items()}
        mapped_zones = df["destination_city"].astype(str).str.strip().str.lower().map(matrix_lower)
        df["verified_zone"] = mapped_zones.fillna(df["zone_id"])
    else:
        df["verified_zone"] = df["zone_id"].apply(
            lambda z: z if str(z).strip().lower() in valid_zones else default_zone
        )

    df["zone_mismatch"] = (
        ~df["zone_id"].astype(str).str.strip().str.lower().isin(valid_zones)
    ) & (len(valid_zones) > 1)

    # د) حساب الوزن المحسوب وتقريبه للأعلى
    df["raw_chargeable_weight"] = df[["total_actual_weight_kg", "total_volumetric_weight_kg"]].max(axis=1)

    if rules.weight_rounding_increment > 0:
        inc = rules.weight_rounding_increment
        df["chargeable_weight"] = np.ceil(df["raw_chargeable_weight"] / inc) * inc
    else:
        df["chargeable_weight"] = df["raw_chargeable_weight"]

    # هـ) مطابقة الأسعار والشرائح
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
    for zone, group_indices in df.groupby(df["verified_zone"].astype(str).str.strip().str.lower()).groups.items():
        zone_rates = rates_df[rates_df["zone_id"] == zone]
        if zone_rates.empty:
            zone_rates = rates_df

        zone_rates = zone_rates.sort_values("max_weight_kg")
        weights = df.loc[group_indices, "chargeable_weight"].values
        group_prices = np.zeros(len(weights))

        for _, rate in zone_rates.iterrows():
            mask = weights <= rate["max_weight_kg"]
            group_prices = np.where((group_prices == 0) & mask, rate["base_price"], group_prices)

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
    is_rto = df["shipment_status"].astype(str).str.lower().isin(["rto", "returned", "مرتجعة", "مرتجع"])

    if rules.surcharges.rto_is_addon:
        df["expected_base_price"] = np.where(
            is_rto, df["expected_base_price"] * (1.0 + rules.surcharges.rto_fee_percentage), df["expected_base_price"]
        )
    else:
        df["expected_base_price"] = np.where(
            is_rto, df["expected_base_price"] * rules.surcharges.rto_fee_percentage, df["expected_base_price"]
        )

    # ز) احتساب الوقود وCOD والضريبة
    df["expected_fuel_fee"] = df["expected_base_price"] * rules.fuel_surcharge_percentage
    df["expected_cod_fee"] = np.where(
        (df["cod_amount"] > 0) & (~is_rto),
        np.maximum(df["cod_amount"] * rules.surcharges.cod_fee_percentage, rules.surcharges.cod_min_fee),
        0.0,
    )
    df["expected_remote_fee"] = np.where(df["is_remote"], rules.surcharges.remote_area_fee, 0.0)

    df["expected_subtotal"] = (
        df["expected_base_price"]
        + df["expected_fuel_fee"]
        + df["expected_cod_fee"]
        + df["expected_remote_fee"]
    )
    df["expected_vat"] = df["expected_subtotal"] * rules.vat_percentage
    df["expected_total"] = (df["expected_subtotal"] + df["expected_vat"]).round(2)

    # معالجة الشحنات المكررة (التكرار الباطل = المستحق 0.0 ريال واسترداد القيمة بالكامل)
    df.loc[df["is_duplicate"], "expected_total"] = 0.0

    df["overcharge"] = (df["billed_amount"] - df["expected_total"]).round(2)

    # ح) فلترة وتقارير المخالفات
    min_thresh = rules.min_overcharge_threshold if rules.min_overcharge_threshold > 0 else 0.1

    flagged = df[
        (df["overcharge"] >= min_thresh)
        | (df["is_duplicate"])
        | (df["zone_mismatch"])
        | (df["has_zero_dims"])
    ].copy()

    def build_evidence(row):
        evidence_list = []
        if row["is_duplicate"]:
            evidence_list.append(f"تكرار غير مشروع بالفاتورة لرقم الشحنة {row['tracking_id']}.")
        if row["zone_mismatch"]:
            evidence_list.append(
                f"اختلاف المنطقة: التسجيل بالفاتورة ({row['zone_id']}) بينما تم تطبيق الشريحة المتاحة بالعقد."
            )
        if row["has_zero_dims"]:
            evidence_list.append("تنبيه: أبعاد الشحنة غير محددة، تم الاحتساب بالوزن الفعلي.")
        if row["overcharge"] >= min_thresh and not row["is_duplicate"]:
            evidence_list.append(
                f"مبالغة بالأسعار: المفلتر بالفاتورة {row['billed_amount']} {rules.currency} | المستحق بالعقد {row['expected_total']} {rules.currency}. "
                f"فروقات مستردة: {row['overcharge']} {rules.currency}."
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
    
