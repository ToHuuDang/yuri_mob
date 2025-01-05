from .connect_database import get_db_connection
from mysql.connector import Error



def get_product_data():
    connection = get_db_connection()
    try:
        cursor = connection.cursor()
        # Ví dụ lấy thông tin sản phẩm
        query = "SELECT id, address, description, internet_cost, is_approve, price, public_electric_cost, title, water_cost, created_by, is_locked FROM room"
        cursor.execute(query)
        products = cursor.fetchall()
        
        # Chuyển đổi dữ liệu thành format phù hợp
        formatted_data = []
        for product in products:
            formatted_data.append({
                 'id': product[0],
                'address': product[1],
                'description': product[2],
                'internet_cost': product[3],
                'is_approve': product[4],
                'price': product[5],
                'public_electric_cost': product[6],
                'title': product[7],
                'water_cost': product[8],
                'created_by': product[9]
            })
        return formatted_data
        
        
    except Error as e:
        print(f"Error: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
