-- Q1: WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS total_spent
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: Three-table join
SELECT c.city, p.category, SUM(o.order_amount) AS total_revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3: Scalar subquery
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: LEFT JOIN
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5: Common Table Expression (CTE)
WITH CustomerTotals AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
AverageSpend AS (
    SELECT AVG(total_spend) AS avg_spend FROM CustomerTotals
)
SELECT ct.customer_id, ct.total_spend
FROM CustomerTotals ct
CROSS JOIN AverageSpend as_avg
WHERE ct.total_spend > as_avg.avg_spend;

-- Q6: Ranking window functions
WITH RankedSpenders AS (
    SELECT 
        c.customer_id, 
        c.city, 
        SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT 
    customer_id, 
    city, 
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_dense_rank,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_row_number
FROM RankedSpenders
ORDER BY city, total_spend DESC;

-- Q7: Running total
WITH DailyRev AS (
    SELECT SUBSTR(order_ts, 1, 10) AS order_day, SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT 
    order_day, 
    daily_revenue,
    SUM(daily_revenue) OVER (ORDER BY order_day) AS cumulative_revenue
FROM DailyRev
ORDER BY order_day;