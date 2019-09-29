--Query for loading data (Tableau)
#standardSQL
  WITH cte3 AS( WITH cte2 AS( with cte1 AS(
      SELECT
        CASE
          WHEN tab_1.cluster IS NULL THEN tab_2.cluster
          ELSE tab_1.cluster
        END cluster,
        CASE
          WHEN tab_1.warehouse IS NULL THEN tab_2.warehouse
          ELSE tab_1.warehouse
        END warehouse,
        CASE
          WHEN tab_1.year IS NULL THEN tab_2.year
          ELSE tab_1.year
        END year,
        CASE
          WHEN tab_1.week IS NULL THEN tab_2.week
          ELSE tab_1.week
        END week,
        tab_1.actual,
        tab_2.forecast
      FROM
        alim_hanif.tab_actual tab_1 FULL
      JOIN
        alim_hanif.tab_forecast tab_2
      ON
        tab_1.cluster = tab_2.cluster
        AND tab_1.warehouse = tab_2.warehouse
        AND tab_1.year = tab_2.year
        AND tab_1.week = tab_2.week )
    SELECT
      cte1.* except(cluster),
      tab_3.* except(cluster),
      CASE
        WHEN cte1.cluster IS NULL THEN tab_3.cluster
        ELSE cte1.cluster
      END cluster
    FROM
      cte1 FULL
    JOIN
      alim_hanif.tab_cluster tab_3
    ON
      cte1.cluster = tab_3.cluster)
  SELECT
    cte2.* except (cluster),
    tab_4.* except(cluster),
    CASE
      WHEN cte2.cluster IS NULL THEN tab_4.cluster
      ELSE tab_4.cluster
    END cluster
  FROM
    cte2 FULL
  JOIN
    alim_hanif.tab_dis_listed_merchant tab_4
  ON
    cte2.cluster = tab_4.cluster)
SELECT
  cte3.* except(warehouse),
  tab_5.* except(warehouse),
  CASE
    WHEN cte3.warehouse IS NULL THEN tab_5.warehouse
    ELSE cte3.warehouse
  END warehouse
FROM
  cte3 FULL
JOIN
  alim_hanif.tab_wh tab_5
ON
  cte3.warehouse = tab_5.warehouse