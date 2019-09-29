#standardSQL
  WITH cte_wh AS ( WITH cte_dian AS (
      WITH plus_dest AS (
      SELECT
        fo.order_id,
        fo.payment_verified_time,
        fo.destination_district_id,
        fo.order_shipping_time,
        fo.order_delivered_time,
        fr.district_name destination_district,
        fr.city_name destination_city,
        fr.province_name destination_province,
        fr.island destination_island,
        fo.origin_district_id,
        fo.shipping_price
      FROM
        acube_2019.aaa_fulfillment_order fo
      LEFT JOIN
        acube_2019.aaa_fulfillment_region fr
      ON
        fo.destination_district_id = fr.district_id
      ORDER BY
        2 ),
      plus_origin AS (
      SELECT
        des.*,
        fr.district_name origin_district,
        fr.city_name origin_city,
        fr.province_name origin_province,
        fr.island origin_island
      FROM
        plus_dest des
      LEFT JOIN
        acube_2019.aaa_fulfillment_region fr
      ON
        des.origin_district_id = fr.district_id ),
      plus_detail AS (
      SELECT
        ori.*,
        fod.product_id,
        fod.product_name,
        fod.quantity,
        fod.shop_id,
        fod.order_status,
        fod.level1_id,
        fod.level2_id,
        fod.level3_id
      FROM
        plus_origin ori
      LEFT JOIN
        acube_2019.aaa_fulfillment_order_dtl fod
      ON
        ori.order_id = fod.order_id ),
      plus_cat AS (
      SELECT
        dtl.*,
        cat.level1_name,
        cat.level2_name,
        cat.level3_name
      FROM
        plus_detail dtl FULL
      JOIN
        acube_2019.aaa_fulfillment_product_category cat
      ON
        dtl.level1_id = cat.level1_id
        AND dtl.level2_id = cat.level2_id
        AND dtl.level3_id = cat.level3_id )
    SELECT
      DATE(plus_cat.payment_verified_time, "Asia/Jakarta") payment_verified_wib_date,
      EXTRACT(YEAR
      FROM
        DATE(plus_cat.payment_verified_time, "Asia/Jakarta")) payment_verified_wib_year,
      EXTRACT(MONTH
      FROM
        DATE(plus_cat.payment_verified_time, "Asia/Jakarta")) payment_verified_wib_month,
      EXTRACT(WEEK
      FROM
        DATE(plus_cat.payment_verified_time, "Asia/Jakarta")) payment_verified_wib_week,
      plus_cat.*,
      FLOOR((EXTRACT(DAYOFYEAR
          FROM
            payment_verified_time) - 1 ) / 7) week_,
      DATE(plus_cat.payment_verified_time),
      tb_cluster.cluster,
      tb_cluster.type
    FROM
      plus_cat
    JOIN
      alim_hanif.data1_cluster tb_cluster
    ON
      LOWER(plus_cat.product_name) = LOWER(tb_cluster.product_name)),
    cte_kab_kot AS (
    SELECT
      *
    FROM
      alim_hanif.data_kab_kot
    WHERE
      warehouse IS NOT NULL)
  SELECT
    cte_dian.*,
    cte_kab_kot.warehouse
  FROM
    cte_dian
  JOIN
    cte_kab_kot
  ON
    cte_dian.destination_city = cte_kab_kot.destination_city),
  new_prov AS (
  SELECT
    *
  FROM
    alim_hanif.data_dict_new_province )
SELECT
  cte_wh.*,
  new_prov.new_origin_city,
  new_prov.new_destination_city
FROM
  cte_wh
LEFT JOIN
  new_prov
ON
  cte_wh.origin_city = new_prov.origin_city
  AND cte_wh.destination_city = new_prov.destination_city