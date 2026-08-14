/**
 * AURA Hybrid AI Recommendation System - Frontend App JavaScript
 * Handles REST API communication, state management, UI rendering, and XAI signal bar visualization.
 */

document.addEventListener('DOMContentLoaded', () => {
    // State Management
    const state = {
        activeUser: 'USER_001',
        category: 'all',
        searchTerm: '',
        page: 1,
        limit: 12,
        selectedProduct: null
    };

    // DOM Elements
    const userSelect = document.getElementById('userSelect');
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const samplePills = document.querySelectorAll('.sample-pill');
    const categoryTabs = document.querySelectorAll('.tab-btn');
    const sectionTitle = document.getElementById('sectionTitle');
    const resultsCount = document.getElementById('resultsCount');
    const productGrid = document.getElementById('productGrid');
    const pagination = document.getElementById('pagination');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');

    // Modal DOM Elements
    const productModal = document.getElementById('productModal');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const modalProductImage = document.getElementById('modalProductImage');
    const modalProductCategory = document.getElementById('modalProductCategory');
    const modalProductName = document.getElementById('modalProductName');
    const modalProductPrice = document.getElementById('modalProductPrice');
    const modalProductRating = document.getElementById('modalProductRating');
    const modalProductDesc = document.getElementById('modalProductDesc');
    const modalProductId = document.getElementById('modalProductId');
    const modalRecsList = document.getElementById('modalRecsList');

    // API Base URL
    const API_BASE = '/api';

    // Helper: Show / Hide Spinner & Error
    function setLoading(isLoading) {
        if (isLoading) {
            loadingSpinner.classList.remove('hidden');
            productGrid.classList.add('hidden');
            errorMessage.classList.add('hidden');
        } else {
            loadingSpinner.classList.add('hidden');
            productGrid.classList.remove('hidden');
        }
    }

    function showError(msg) {
        loadingSpinner.classList.add('hidden');
        productGrid.classList.add('hidden');
        errorMessage.classList.remove('hidden');
        errorText.textContent = msg || 'An error occurred while connecting to the recommendation service.';
    }

    // Helper: Image Path Formatter
    function getImageUrl(imagePath) {
        if (!imagePath) return 'https://via.placeholder.com/300x200?text=No+Image';
        if (imagePath.startsWith('http://') || imagePath.startsWith('https://')) return imagePath;
        const filename = imagePath.split(/[\/\\]/).pop();
        return `${API_BASE}/images/${filename}`;
    }

    // 1. Fetch & Render Product Catalog
    async function loadProducts() {
        setLoading(true);
        try {
            const params = new URLSearchParams({
                page: state.page,
                limit: state.limit,
                category: state.category
            });

            if (state.searchTerm) {
                params.append('search', state.searchTerm);
            }

            const response = await fetch(`${API_BASE}/products?${params.toString()}`);
            const data = await response.json();

            if (!data.success) {
                showError(data.error?.message || 'Failed to fetch products');
                return;
            }

            renderProductGrid(data.data);
            renderPagination(data.pagination);

            if (state.searchTerm) {
                sectionTitle.innerHTML = `<i class="fa-solid fa-magnifying-glass"></i> Keyword Results for "${state.searchTerm}"`;
            } else if (state.category !== 'all') {
                sectionTitle.innerHTML = `<i class="fa-solid fa-filter"></i> Category: ${state.category}`;
            } else {
                sectionTitle.innerHTML = `<i class="fa-solid fa-store"></i> Product Catalog`;
            }

            resultsCount.textContent = `${data.pagination.total_items} Products`;
            setLoading(false);
        } catch (err) {
            showError('Unable to connect to Flask REST API server.');
        }
    }

    // 2. Execute NLP Semantic Vector Search
    async function executeSemanticSearch(queryText) {
        if (!queryText || !queryText.trim()) return;

        setLoading(true);
        try {
            const response = await fetch(`${API_BASE}/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: queryText, top_k: 12 })
            });

            const data = await response.json();
            if (!data.success) {
                showError(data.error?.message || 'Search execution failed');
                return;
            }

            renderProductGrid(data.results);
            pagination.innerHTML = ''; // No pagination for semantic search results
            sectionTitle.innerHTML = `<i class="fa-solid fa-sparkles"></i> NLP Semantic Results for "${queryText}"`;
            resultsCount.textContent = `${data.results.length} Matches`;
            setLoading(false);
        } catch (err) {
            showError('Error executing semantic vector search.');
        }
    }

    // 3. Render Product Cards Grid
    function renderProductGrid(products) {
        productGrid.innerHTML = '';

        if (!products || products.length === 0) {
            productGrid.innerHTML = `
                <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-muted);">
                    <i class="fa-solid fa-box-open" style="font-size: 36px; margin-bottom: 10px;"></i>
                    <p>No products found matching your criteria.</p>
                </div>
            `;
            return;
        }

        products.forEach(p => {
            const card = document.createElement('div');
            card.className = 'product-card';

            const imgUrl = getImageUrl(p.image_url || p.image_path);
            const price = typeof p.price === 'number' ? `$${p.price.toFixed(2)}` : '$0.00';
            const rating = typeof p.rating === 'number' ? p.rating.toFixed(1) : '4.5';
            const simScore = p.similarity_score ? ` &bull; Match: ${(p.similarity_score * 100).toFixed(1)}%` : '';

            card.innerHTML = `
                <div class="card-img-wrapper">
                    <span class="category-badge">${p.category || 'General'}</span>
                    <img src="${imgUrl}" alt="${p.product_name}" class="card-img" onerror="this.src='https://via.placeholder.com/300x200?text=Catalog+Product'">
                </div>
                <div class="card-content">
                    <h3 class="card-title">${p.product_name}</h3>
                    <p class="card-desc">${p.description || 'Quality product available in catalog.'}</p>
                    <div class="card-meta">
                        <span class="card-price">${price}</span>
                        <span class="card-rating"><i class="fa-solid fa-star"></i> ${rating}${simScore}</span>
                    </div>
                    <div class="card-actions">
                        <button class="btn-primary view-details-btn" data-id="${p.product_id}"><i class="fa-solid fa-circle-info"></i> Details & Recs</button>
                    </div>
                </div>
            `;

            card.querySelector('.view-details-btn').addEventListener('click', () => {
                openProductModal(p);
            });

            productGrid.appendChild(card);
        });
    }

    // 4. Render Pagination Controls
    function renderPagination(pInfo) {
        pagination.innerHTML = '';
        if (pInfo.total_pages <= 1) return;

        for (let i = 1; i <= pInfo.total_pages; i++) {
            const btn = document.createElement('button');
            btn.className = `page-btn ${i === pInfo.page ? 'active' : ''}`;
            btn.textContent = i;
            btn.addEventListener('click', () => {
                state.page = i;
                loadProducts();
            });
            pagination.appendChild(btn);
        }
    }

    // 5. Open Product Modal & Fetch Hybrid Recommendations with XAI
    async function openProductModal(product) {
        state.selectedProduct = product;

        modalProductImage.src = getImageUrl(product.image_url || product.image_path);
        modalProductCategory.textContent = product.category || 'General';
        modalProductName.textContent = product.product_name;
        modalProductPrice.textContent = typeof product.price === 'number' ? `$${product.price.toFixed(2)}` : '$0.00';
        modalProductRating.innerHTML = `<i class="fa-solid fa-star"></i> ${typeof product.rating === 'number' ? product.rating.toFixed(1) : '4.5'}`;
        modalProductDesc.textContent = product.description || 'Product details...';
        modalProductId.textContent = `ID: ${product.product_id}`;

        modalRecsList.innerHTML = '<div class="spinner" style="width:30px;height:30px;"></div><p style="text-align:center;font-size:12px;">Computing Hybrid Fusion & XAI Score Breakdown...</p>';

        productModal.classList.remove('hidden');

        // Fetch Hybrid Recs via API
        try {
            const payload = {
                user_id: state.activeUser,
                product_id: product.product_id,
                query: state.searchTerm || null,
                top_k: 5
            };

            const response = await fetch(`${API_BASE}/recommendations/hybrid`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (!data.success || !data.recommendations) {
                modalRecsList.innerHTML = '<p style="color:var(--text-muted);">No recommendations available for this item.</p>';
                return;
            }

            renderHybridRecommendations(data.recommendations);
        } catch (err) {
            modalRecsList.innerHTML = '<p style="color:var(--accent-rose);">Failed to load hybrid recommendations.</p>';
        }
    }

    // 6. Render Hybrid Recommendations & Visual XAI Signal Contribution Bars
    function renderHybridRecommendations(recs) {
        modalRecsList.innerHTML = '';

        if (recs.length === 0) {
            modalRecsList.innerHTML = '<p style="color:var(--text-muted);">No recommendations returned.</p>';
            return;
        }

        recs.forEach(rec => {
            const item = document.createElement('div');
            item.className = 'rec-item-card';

            const score = typeof rec.final_hybrid_score === 'number' ? rec.final_hybrid_score.toFixed(4) : '0.0000';

            // XAI Contributions
            const cContent = (rec.content_contribution || 0).toFixed(2);
            const cCollab = (rec.collaborative_contribution || 0).toFixed(2);
            const cNlp = (rec.nlp_contribution || 0).toFixed(2);
            const cImage = (rec.image_contribution || 0).toFixed(2);

            // Convert contribution scores to percentage width (max expected single contribution ~0.35)
            const wContent = Math.min(100, Math.max(5, (rec.content_contribution / 0.35) * 100));
            const wCollab = Math.min(100, Math.max(5, (rec.collaborative_contribution / 0.35) * 100));
            const wNlp = Math.min(100, Math.max(5, (rec.nlp_contribution / 0.35) * 100));
            const wImage = Math.min(100, Math.max(5, (rec.image_contribution / 0.35) * 100));

            item.innerHTML = `
                <div class="rec-item-header">
                    <span class="rec-item-title"><i class="fa-solid fa-cube"></i> ${rec.product_name}</span>
                    <span class="rec-score-badge">Score: ${score}</span>
                </div>
                
                <div class="xai-bars-container">
                    <div class="xai-bar-group">
                        <div class="xai-bar-label"><span>Content</span> <span>+${cContent}</span></div>
                        <div class="xai-progress-track"><div class="xai-progress-fill fill-content" style="width: ${wContent}%"></div></div>
                    </div>
                    <div class="xai-bar-group">
                        <div class="xai-bar-label"><span>Collab Taste</span> <span>+${cCollab}</span></div>
                        <div class="xai-progress-track"><div class="xai-progress-fill fill-collab" style="width: ${wCollab}%"></div></div>
                    </div>
                    <div class="xai-bar-group">
                        <div class="xai-bar-label"><span>NLP Vector</span> <span>+${cNlp}</span></div>
                        <div class="xai-progress-track"><div class="xai-progress-fill fill-nlp" style="width: ${wNlp}%"></div></div>
                    </div>
                    <div class="xai-bar-group">
                        <div class="xai-bar-label"><span>CNN Image</span> <span>+${cImage}</span></div>
                        <div class="xai-progress-track"><div class="xai-progress-fill fill-image" style="width: ${wImage}%"></div></div>
                    </div>
                </div>

                <div class="rec-explanation-text">
                    <i class="fa-solid fa-circle-info" style="color:var(--primary)"></i> <strong>Why Recommended:</strong> ${rec.explanation || 'Recommended based on weighted signal similarity.'}
                </div>
            `;

            modalRecsList.appendChild(item);
        });
    }

    // Event Listeners
    userSelect.addEventListener('change', (e) => {
        state.activeUser = e.target.value;
        if (state.selectedProduct && !productModal.classList.contains('hidden')) {
            openProductModal(state.selectedProduct); // Re-fetch recs for new user profile
        }
    });

    searchBtn.addEventListener('click', () => {
        const query = searchInput.value.trim();
        if (query) {
            state.searchTerm = query;
            executeSemanticSearch(query);
        } else {
            state.searchTerm = '';
            loadProducts();
        }
    });

    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            searchBtn.click();
        }
    });

    samplePills.forEach(pill => {
        pill.addEventListener('click', () => {
            const query = pill.getAttribute('data-query');
            searchInput.value = query;
            state.searchTerm = query;
            executeSemanticSearch(query);
        });
    });

    categoryTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            categoryTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            state.category = tab.getAttribute('data-category');
            state.page = 1;
            state.searchTerm = '';
            searchInput.value = '';
            loadProducts();
        });
    });

    closeModalBtn.addEventListener('click', () => {
        productModal.classList.add('hidden');
    });

    productModal.addEventListener('click', (e) => {
        if (e.target === productModal) {
            productModal.classList.add('hidden');
        }
    });

    // Initial Load
    loadProducts();
});
