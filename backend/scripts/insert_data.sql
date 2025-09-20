-- Insert Users
INSERT INTO users (name, email, password, age, location) VALUES
('Alice Johnson', 'alice@example.com', 'password123', 25, 'New York'),
('Bob Smith', 'bob@example.com', 'securepass', 30, 'Los Angeles'),
('Charlie Brown', 'charlie@example.com', 'mypassword', 22, 'Chicago'),
('David White', 'david@example.com', 'pass1234', 28, 'Houston'),
('Emily Davis', 'emily@example.com', 'emilypass', 35, 'San Francisco');

-- Insert Products
INSERT INTO products (name, category, description, price, rating, stock, image_url) VALUES
('iPhone 14 Pro', 'Electronics', 'Latest Apple smartphone with advanced camera', 999.99, 4.8, 50, 'iphone14.jpg'),
('Samsung Galaxy S23', 'Electronics', 'High-performance Android phone', 899.99, 4.5, 30, 'galaxy_s23.jpg'),
('Nike Running Shoes', 'Clothing', 'Lightweight and comfortable running shoes', 120.00, 4.6, 20, 'nike_shoes.jpg'),
('MacBook Air M2', 'Electronics', 'Powerful laptop with M2 chip', 1249.99, 4.7, 15, 'macbook_air.jpg'),
("Levi's Denim Jacket", 'Fashion', 'Classic denim jacket for all seasons', 89.99, 4.3, 40, 'levis_jacket.jpg');

-- Insert Reviews
INSERT INTO reviews (user_id, product_id, rating, review_text) VALUES
(1, 1, 5, 'Absolutely love the iPhone 14 Pro! The camera is amazing.'),
(2, 2, 4, 'Samsung Galaxy S23 is great, but battery life could be better.'),
(3, 3, 5, 'Super comfortable running shoes! Nike never disappoints.'),
(4, 4, 5, 'MacBook Air M2 is super fast and lightweight.'),
(5, 5, 4, 'Nice denim jacket, but the size runs a bit small.');

-- Insert Transactions
INSERT INTO transactions (user_id, product_id, quantity, total_price) VALUES
(1, 1, 1, 999.99),
(2, 2, 1, 899.99),
(3, 3, 2, 240.00),
(4, 4, 1, 1249.99),
(5, 5, 1, 89.99);

-- Insert Browsing History
INSERT INTO browsing_history (user_id, product_id, action) VALUES
(1, 2, 'viewed'),
(2, 3, 'clicked'),
(3, 4, 'added_to_cart'),
(4, 1, 'viewed'),
(5, 2, 'clicked');

-- Insert Social Media Data
INSERT INTO social_media_data (user_id, product_id, engagement_score) VALUES
(1, 3, 8.5),
(2, 1, 9.2),
(3, 4, 7.8),
(4, 5, 8.9),
(5, 2, 6.4);

-- Insert Wishlist Items
INSERT INTO wishlist (user_id, product_id) VALUES
(1, 3),
(2, 1),
(3, 5),
(4, 2),
(5, 4);
