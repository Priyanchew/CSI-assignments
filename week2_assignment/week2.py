class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        print(" -> ".join(elements) if elements else "Empty List")

    def delete_nth(self, n):
        if not self.head:
            raise IndexError("Cannot delete from an empty list.")
        if n <= 0:
            raise IndexError("Index out of range. n should be >= 1.")
        if n == 1:
            deleted = self.head
            self.head = self.head.next
            deleted.next = None
            return
        current = self.head
        count = 1
        while current and count < n - 1:
            current = current.next
            count += 1
        if not current or not current.next:
            raise IndexError(f"Index out of range. There is no node at position {n}.")
        to_delete = current.next
        current.next = to_delete.next
        to_delete.next = None

if __name__ == "__main__":
    ll = LinkedList()
    ll.print_list()
    for i in range(1, 6):
        ll.add_node(i)
    ll.print_list()
    try:
        ll.delete_nth(1)
        ll.print_list()
        ll.delete_nth(3)
        ll.print_list()
        ll.delete_nth(10)
    except IndexError as e:
        print(f"Error: {e}")
    try:
        while True:
            ll.delete_nth(1)
            ll.print_list()
    except IndexError as e:
        print(f"Finished deleting all nodes: {e}")
